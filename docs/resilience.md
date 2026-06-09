# Execra Pipeline Resilience

> **Document scope:** Chaos engineering methodology, failure modes discovered in
> `tests/chaos/test_pipeline_resilience.py`, and the recovery strategies built
> into `ExecraPipeline`.

---

## Table of Contents

1. [Overview](#overview)
2. [Chaos Test Architecture](#chaos-test-architecture)
3. [Failure Modes & Recovery Strategies](#failure-modes--recovery-strategies)
   - [LLM Timeout (20 %)](#1-llm-client-timeout-20-)
   - [Redis Cache Miss (50 %)](#2-redis-cache-miss-50-)
   - [Perception Frame Drop (30 %)](#3-perception-frame-drop-30-)
   - [Combined / Cascading Faults](#4-combined--cascading-faults)
4. [Degraded-Mode Behaviour](#degraded-mode-behaviour)
5. [Observability](#observability)
6. [Running the Chaos Suite](#running-the-chaos-suite)
7. [Extending the Chaos Suite](#extending-the-chaos-suite)
8. [Design Principles](#design-principles)

---

## Overview

Execra is a real-time, multimodal execution intelligence layer.  Its pipeline
runs continuously in the background and must deliver guidance even when
individual subsystems are unhealthy.  Transient network failures, cold cache
states, and dropped sensor frames are normal operating conditions — the pipeline
must degrade gracefully rather than crash.

The chaos suite in `tests/chaos/test_pipeline_resilience.py` validates this by
running `ExecraPipeline` for **60 seconds** while the `ChaosMonkey` fixture
injects three independent fault types simultaneously.

---

## Chaos Test Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         ChaosMonkey                              │
│                                                                  │
│   wrap_llm()    ──►  20 % of complete() calls → TimeoutError    │
│   wrap_redis()  ──►  50 % of get()     calls → None (miss)      │
│   wrap_queue()  ──►  30 % of put()     calls → silently dropped │
└──────────────────────────────────────────────────────────────────┘
              │                │                │
              ▼                ▼                ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
   │  LLMClient   │  │ RedisClient  │  │  Perception Queue    │
   │  (chaotic)   │  │  (chaotic)   │  │  (chaotic)           │
   └──────────────┘  └──────────────┘  └──────────────────────┘
              │                │                │
              └────────────────┴────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   ExecraPipeline     │
                    │   ._tick()           │
                    │   .guidance_delivered│
                    │   .error_log         │
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  GuidanceDispatcher  │
                    │  (registered channel │
                    │   captures output)   │
                    └──────────────────────┘
```

The `ChaosMonkey` uses a **seeded `random.Random`** instance (`seed=2026`) so
every test run produces an identical fault sequence, making failures
fully reproducible.

---

## Failure Modes & Recovery Strategies

### 1. LLM Client Timeout (20 %)

| Attribute | Detail |
|-----------|--------|
| **Trigger** | `TimeoutError` raised inside `BaseLLMClient.complete()` |
| **Probability** | 20 % of all `complete()` invocations |
| **Root causes** | Network congestion · OpenAI / Gemini API rate-limit · Model inference latency spike |

#### What happens without resilience

The unhandled `TimeoutError` propagates up through `_call_llm()` →
`_tick()` → `run()` and crashes the pipeline coroutine, leaving the user
with no guidance for the rest of the session.

#### Recovery strategy implemented

```python
# core/pipeline (ExecraPipeline._call_llm)
try:
    raw = await self.llm_client.complete(prompt)
    return self._make_instruction(raw, confidence=0.85, source="llm")
except TimeoutError as exc:
    self._log_error("llm_timeout", str(exc), step=self._step)
    # Graceful degradation: deterministic fallback instruction
    return self._make_instruction(
        "LLM unavailable — continue with last known safe action.",
        confidence=0.30,       # ← marked degraded
        source="fallback",
    )
```

Recovery steps in order:

1. **Catch** `TimeoutError` inside `_call_llm()`; never let it escape.
2. **Log** a structured `{"kind": "llm_timeout", ...}` entry.
3. **Return** a fallback `GuidanceInstruction` with `confidence < 0.5` so
   callers (and the trust scorer) know the output is degraded.
4. The `@retry(max_retries=3, base_delay=2)` decorator on the real
   `BaseLLMClient.complete()` already retries transient failures before the
   chaos wrapper ever sees them; the fallback only fires if all retries fail.

---

### 2. Redis Cache Miss (50 %)

| Attribute | Detail |
|-----------|--------|
| **Trigger** | `redis.get()` returns `None` |
| **Probability** | 50 % of reads |
| **Root causes** | Cache cold-start · TTL expiry · Redis node restart · Eviction under memory pressure |

#### What happens without resilience

A cache miss is not an error per se, but if the pipeline treats `None` as
authoritative (i.e. never falls through to the LLM), guidance is silently
omitted for half of all ticks.

#### Recovery strategy implemented

```python
async def _tick(self) -> None:
    frame_text = await self._drain_perception_queue()
    cached     = await self._redis_get(cache_key)

    if cached is not None:
        instruction = cached          # fast path
    else:
        instruction = await self._call_llm(frame_text)  # slow path fallback
```

The pipeline treats Redis as a **read-through cache**, not a source of truth.
A miss simply falls through to a fresh LLM call.  Redis errors (connection
refused, timeout) are also caught in `_redis_get()` and logged as
`{"kind": "redis_get", ...}` before falling through to the LLM.

**Additional hardening recommendations:**

- Set a short `socket_timeout` (e.g. 200 ms) on the Redis client so a
  slow node does not stall the tick loop.
- Use `redis.asyncio` with `retry_on_timeout=True` for automatic reconnect.
- Consider a local in-process LRU cache (`functools.lru_cache` or
  `cachetools.TTLCache`) as a secondary layer between Redis and the LLM.

---

### 3. Perception Frame Drop (30 %)

| Attribute | Detail |
|-----------|--------|
| **Trigger** | `asyncio.Queue.put()` / `put_nowait()` silently discards the frame |
| **Probability** | 30 % of enqueued frames |
| **Root causes** | Camera hardware stall · `mss` screenshot throttling · OS scheduler jitter · Queue back-pressure |

#### What happens without resilience

If the pipeline blocks waiting for a frame that was dropped, the tick loop
stalls indefinitely and guidance delivery stops.

#### Recovery strategy implemented

```python
async def _drain_perception_queue(self) -> str:
    texts: list[str] = []
    while True:
        try:
            frame = self.perception_queue.get_nowait()
            texts.append(frame.get("text", ""))
        except asyncio.QueueEmpty:
            break
    return " ".join(texts) if texts else "no frame data"   # ← never blocks
```

Key design choices:

- **Non-blocking drain** via `get_nowait()` — if the queue is empty (because
  all recent frames were dropped) the pipeline tick continues with
  `"no frame data"` as the context string.
- The LLM prompt still executes, producing guidance based on session context
  rather than the current frame.  Confidence is not penalised for missing
  frames — the LLM is expected to handle thin context gracefully.
- The `PerceptionBus` producers (`ScreenCapture`, `CameraFeed`) already run
  in background threads at a configurable FPS (`SCREEN_CAPTURE_FPS`), so
  the next frame arrives within one FPS interval even after a run of drops.

---

### 4. Combined / Cascading Faults

When all three fault types fire in the same tick (probability ≈ 0.20 × 0.50 ×
0.30 = 3 %):

| Phase | Fault | Recovery |
|-------|-------|----------|
| Perception | Frame dropped | `_drain_perception_queue` returns `"no frame data"` |
| Cache | Redis miss | Falls through to LLM |
| LLM | TimeoutError | Returns fallback instruction (`confidence=0.30`) |

The pipeline still delivers a guidance instruction — degraded, but present.
No exception propagates to the caller.

---

## Degraded-Mode Behaviour

The pipeline communicates degradation through the `confidence` field of
`GuidanceInstruction`:

| Mode | `confidence` range | `source` field | Meaning |
|------|--------------------|----------------|---------|
| Normal | `≥ 0.50` | `["llm"]` | Fresh LLM response |
| LLM timeout fallback | `0.30` | `["fallback"]` | Static safe-action message |
| LLM error fallback | `0.20` | `["fallback"]` | Generic caution message |

Downstream consumers (dashboard, WebSocket clients) should:

1. Check `confidence < 0.5` before auto-applying guidance.
2. Display a visual indicator (e.g. amber badge) when `source == ["fallback"]`.
3. Optionally suppress fallback guidance if `confidence < 0.25` and the
   previous instruction was delivered within the last 5 seconds.

The `TrustScorer` already uses `llm_confidence` as input; passing the
`confidence` field directly keeps the trust pipeline consistent.

---

## Observability

Every subsystem failure is recorded in `pipeline.error_log` as a plain
`dict` with guaranteed keys:

```python
{
    "kind"   : "llm_timeout",      # or "llm_error", "redis_get"
    "message": "<exception text>",
    "step"   : 42,                 # pipeline tick number
    # optional extra context keys (e.g. "key" for redis_get)
}
```

All entries are also emitted at `WARNING` level via Python's standard
`logging` module under the logger name `execra.chaos` (test-visible) and
`ExecraPipeline` (production).  Integrate with your log aggregator
(e.g. Loki, CloudWatch, Datadog) by adding a structured JSON formatter.

**Recommended alert thresholds:**

| Metric | Warning | Critical |
|--------|---------|----------|
| `llm_timeout` rate (per minute) | > 10 % | > 40 % |
| `redis_get` error rate | > 5 % | > 20 % |
| Guidance `confidence` p50 | < 0.6 | < 0.4 |

---

## Running the Chaos Suite

### Prerequisites

```powershell
# 1. Install dev dependencies (run once)
pip install -r requirements-dev.txt --break-system-packages

# 2. Set a minimal .env (tests mock LLM/Redis, but config.py validates env)
@"
LLM_BACKEND=openai
OPENAI_API_KEY=sk-test-chaos
GEMINI_API_KEY=AItest-chaos
ENCRYPTION_KEY=chaos-test-key-32-bytes-padding!!
"@ | Set-Content .env
```

### Run only the chaos tests

```powershell
pytest tests/chaos/test_pipeline_resilience.py -v --timeout=120
```

### Run with coverage

```powershell
pytest tests/chaos/ -v --timeout=120 --cov=core --cov-report=term-missing
```

### Run in parallel (faster CI)

```powershell
pip install pytest-xdist --break-system-packages
pytest tests/chaos/ -n 4 --timeout=120
```

### Expected output (summary)

```
tests/chaos/test_pipeline_resilience.py::TestPipelineResilience::test_pipeline_never_crashes_under_chaos PASSED
tests/chaos/test_pipeline_resilience.py::TestPipelineResilience::test_guidance_still_delivered_under_chaos PASSED
tests/chaos/test_pipeline_resilience.py::TestPipelineResilience::test_all_errors_are_logged PASSED
tests/chaos/test_pipeline_resilience.py::TestPipelineResilience::test_llm_timeout_triggers_fallback_guidance PASSED
... (12 tests total)

CHAOS SUMMARY
  LLM timeouts injected : ~24  (≈20 % of 120 ticks)
  Redis misses injected  : ~60  (≈50 % of 120 ticks)
  Frames dropped         : ~72  (≈30 % of ~240 produced frames)
  Guidance delivered     : ≥ 90 (≥75 % of 120 ticks)
  Errors logged          : ≥ 24 (all timeouts captured)
```

---

## Extending the Chaos Suite

Add new fault types by extending `ChaosMonkey`:

```python
def wrap_sqlite(self, base_engine):
    """Inject 10 % SQLite lock errors on context_engine reads."""
    monkey = self

    class ChaoticContextEngine:
        async def get_context(self, session_id):
            if monkey._rng.random() < 0.10:
                monkey.db_errors_injected += 1
                raise aiosqlite.OperationalError("chaos: database is locked")
            return await base_engine.get_context(session_id)
    return ChaoticContextEngine()
```

Then add a corresponding `wrap_sqlite` call in the `chaos_pipeline` fixture
and a new assertion test.

Suggested additional fault vectors:

| Fault | Probability | Recovery |
|-------|-------------|----------|
| OCR engine raises `RuntimeError` | 15 % | Return empty string for screen text |
| Object detector returns empty list | 25 % | Skip plugin rule evaluation |
| SQLite `OperationalError` on context read | 10 % | Use in-memory fallback context |
| WebSocket send raises `ConnectionResetError` | 5 % | Log and re-register channel |

---

## Design Principles

1. **Never let subsystem exceptions escape the pipeline loop.**  Every `await`
   that touches an external resource is wrapped in `try/except`.

2. **Degrade gracefully, not silently.**  Fallback guidance is always
   delivered with `confidence < 0.5` so consumers can distinguish it from
   normal output.

3. **Log everything, assert on logs.**  Structured error entries allow the
   test suite to verify that faults are not swallowed — `len(error_log) ==
   chaos_monkey.timeouts_injected` is a first-class assertion.

4. **Reproducible chaos.**  Seeded `random.Random` means a failing CI run
   can be replayed exactly with the same fault sequence.

5. **Observe rates, not just pass/fail.**  Statistical assertions
   (`|observed_rate - expected| < 0.15`) catch `ChaosMonkey` configuration
   drift and biased random number generators early.
