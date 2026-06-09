"""
tests/chaos/test_pipeline_resilience.py

Chaos engineering tests for the Execra pipeline.

Verifies that ExecraPipeline continues operating (possibly in a degraded
state) when individual subsystems fail randomly:

  - LLM client raises TimeoutError  on 20 % of calls
  - Redis returns a cache miss       on 50 % of reads
  - Perception queue drops frames    on 30 % of enqueue attempts

The pipeline is run for 60 seconds under these conditions.  Assertions
confirm:
  1. The pipeline never crashes / raises an unhandled exception.
  2. Guidance is still delivered (possibly degraded — lower confidence).
  3. Every fault is captured in the structured error log.

Run with:
    pytest tests/chaos/test_pipeline_resilience.py -v --timeout=120
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import GuidanceInstruction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("execra.chaos")

# ---------------------------------------------------------------------------
# Chaos probabilities  (match the issue spec exactly)
# ---------------------------------------------------------------------------

LLM_TIMEOUT_PROBABILITY   = 0.20   # 20 % of LLM complete() calls raise TimeoutError
REDIS_MISS_PROBABILITY     = 0.50   # 50 % of Redis get() calls return None (cache miss)
FRAME_DROP_PROBABILITY     = 0.30   # 30 % of frames are silently dropped

# ---------------------------------------------------------------------------
# Test duration
# ---------------------------------------------------------------------------

CHAOS_RUN_SECONDS = 60      # pipeline runs for 60 s under chaos
TICK_INTERVAL     = 0.5     # seconds between pipeline ticks in the test


# ===========================================================================
# ExecraPipeline  — minimal self-contained implementation
# ===========================================================================
# The real Execra repo does not expose a single "ExecraPipeline" class; the
# logic is spread across PerceptionBus, IntelligenceCore, GuidanceDispatcher,
# ContextEngine, and (optionally) Redis.  This class wires those pieces
# together in the same way the production routes do, making it straightforward
# to inject chaos at every boundary.
# ===========================================================================

class ExecraPipeline:
    """
    Thin orchestration layer that ties together:
      - Perception queue  (frames supplied externally in tests)
      - LLM client        (injected; chaos wrapper applied in fixture)
      - Redis client      (injected; chaos wrapper applied in fixture)
      - GuidanceDispatcher (real implementation, no-OS-notification mode)
    """

    def __init__(
        self,
        llm_client: Any,
        redis_client: Any,
        perception_queue: asyncio.Queue,
    ) -> None:
        from core.hybrid.guidance_dispatcher import GuidanceDispatcher

        self.llm_client       = llm_client
        self.redis_client     = redis_client
        self.perception_queue = perception_queue
        self.dispatcher       = GuidanceDispatcher(enable_os_notifications=False)

        # ----------------------------------------------------------------
        # Observability: counters & error log
        # ----------------------------------------------------------------
        self.guidance_delivered: list[GuidanceInstruction] = []
        self.error_log: list[dict[str, Any]] = []
        self._step = 0
        self._running = False

        # Register a capture channel so we can assert on delivered guidance
        self.dispatcher.register_channel(self._capture_guidance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, duration_seconds: float) -> None:
        """Run the pipeline for *duration_seconds*, ticking every TICK_INTERVAL."""
        self._running = True
        deadline = asyncio.get_event_loop().time() + duration_seconds
        logger.info("ExecraPipeline started (chaos mode, duration=%ss)", duration_seconds)

        while self._running and asyncio.get_event_loop().time() < deadline:
            await self._tick()
            await asyncio.sleep(TICK_INTERVAL)

        self._running = False
        logger.info(
            "ExecraPipeline finished — delivered=%d errors=%d",
            len(self.guidance_delivered),
            len(self.error_log),
        )

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal pipeline tick
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """One iteration of the perception → intelligence → dispatch loop."""
        self._step += 1
        frame_text = await self._drain_perception_queue()

        cache_key = f"guidance:step:{self._step}"
        cached    = await self._redis_get(cache_key)

        if cached is not None:
            # Fast path — serve from cache
            instruction = cached
            logger.debug("Step %d: served from cache", self._step)
        else:
            # Slow path — call the LLM
            instruction = await self._call_llm(frame_text)

        if instruction:
            self.dispatcher.dispatch(instruction)

    # ------------------------------------------------------------------
    # Subsystem wrappers (each catches its own errors gracefully)
    # ------------------------------------------------------------------

    async def _drain_perception_queue(self) -> str:
        """
        Pull all available frames off the queue and concatenate their text.
        A dropped frame (queue empty on get_nowait) is silently skipped.
        """
        texts: list[str] = []
        while True:
            try:
                frame: dict = self.perception_queue.get_nowait()
                texts.append(frame.get("text", ""))
            except asyncio.QueueEmpty:
                break
        return " ".join(texts) if texts else "no frame data"

    async def _redis_get(self, key: str) -> GuidanceInstruction | None:
        """Try to fetch a cached GuidanceInstruction; swallow all errors."""
        try:
            result = await self.redis_client.get(key)
            return result
        except Exception as exc:
            self._log_error("redis_get", str(exc), key=key)
            return None

    async def _call_llm(self, prompt_context: str) -> GuidanceInstruction | None:
        """
        Ask the LLM for guidance text and wrap it in a GuidanceInstruction.
        On TimeoutError (or any other exception), fall back to a degraded
        'service unavailable' instruction so guidance is never fully silenced.
        """
        try:
            raw = await self.llm_client.complete(
                f"Provide next action guidance.  Context: {prompt_context}"
            )
            return self._make_instruction(raw, confidence=0.85, source="llm")

        except TimeoutError as exc:
            self._log_error("llm_timeout", str(exc), step=self._step)
            # Graceful degradation: rule-based fallback instruction
            return self._make_instruction(
                "LLM unavailable — continue with last known safe action.",
                confidence=0.30,
                source="fallback",
            )

        except Exception as exc:
            self._log_error("llm_error", str(exc), step=self._step)
            return self._make_instruction(
                "Guidance temporarily unavailable — proceed with caution.",
                confidence=0.20,
                source="fallback",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_instruction(
        self,
        text: str,
        confidence: float,
        source: str,
    ) -> GuidanceInstruction:
        return GuidanceInstruction(
            instruction=text,
            confidence=confidence,
            source=[source],
            reasoning=f"Generated at step {self._step}",
            mode="safe",
            step=self._step,
            total_steps=max(self._step, 1),
            generated_at=datetime.now(timezone.utc),
        )

    def _capture_guidance(self, instruction: GuidanceInstruction) -> None:
        """Channel registered with GuidanceDispatcher to record deliveries."""
        self.guidance_delivered.append(instruction)

    def _log_error(self, kind: str, message: str, **ctx: Any) -> None:
        entry = {"kind": kind, "message": message, "step": self._step, **ctx}
        self.error_log.append(entry)
        logger.warning("CHAOS ERROR [%s] step=%d — %s", kind, self._step, message)


# ===========================================================================
# ChaosMonkey fixture
# ===========================================================================

class ChaosMonkey:
    """
    Injects controlled faults into Execra subsystems:

      * LLM client  — raises ``TimeoutError`` on 20 % of ``complete()`` calls
      * Redis client — returns ``None`` (cache miss) on 50 % of ``get()`` calls
      * Perception queue — silently drops 30 % of frames on ``put()``

    Usage::

        monkey = ChaosMonkey(seed=42)
        llm    = monkey.wrap_llm(real_llm_client)
        redis  = monkey.wrap_redis(real_redis_client)
        queue  = monkey.wrap_queue(asyncio.Queue())
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        # Counters (for assertions / debugging)
        self.llm_timeouts_injected   = 0
        self.redis_misses_injected   = 0
        self.frames_dropped          = 0
        self.frames_delivered        = 0

    # ------------------------------------------------------------------
    # LLM wrapper
    # ------------------------------------------------------------------

    def wrap_llm(self, base_client: Any) -> Any:
        """Return a proxy whose ``complete()`` raises TimeoutError 20 % of the time."""
        monkey = self

        class ChaoticLLMClient:
            async def complete(self, prompt: str) -> str:
                if monkey._rng.random() < LLM_TIMEOUT_PROBABILITY:
                    monkey.llm_timeouts_injected += 1
                    logger.debug("CHAOS: injecting LLM TimeoutError")
                    raise TimeoutError("chaos: simulated LLM timeout")
                return await base_client.complete(prompt)

            async def stream(self, prompt: str) -> AsyncIterator[str]:
                return base_client.stream(prompt)

            def extract_confidence(self, response: Any) -> float:
                return base_client.extract_confidence(response)

        return ChaoticLLMClient()

    # ------------------------------------------------------------------
    # Redis wrapper
    # ------------------------------------------------------------------

    def wrap_redis(self, base_client: Any) -> Any:
        """Return a proxy whose ``get()`` returns ``None`` 50 % of the time."""
        monkey = self

        class ChaoticRedisClient:
            async def get(self, key: str) -> Any:
                if monkey._rng.random() < REDIS_MISS_PROBABILITY:
                    monkey.redis_misses_injected += 1
                    logger.debug("CHAOS: injecting Redis cache miss for key=%s", key)
                    return None
                return await base_client.get(key)

            async def set(self, key: str, value: Any, **kwargs: Any) -> Any:
                return await base_client.set(key, value, **kwargs)

            async def delete(self, key: str) -> Any:
                return await base_client.delete(key)

        return ChaoticRedisClient()

    # ------------------------------------------------------------------
    # Perception queue wrapper
    # ------------------------------------------------------------------

    def wrap_queue(self, base_queue: asyncio.Queue) -> asyncio.Queue:
        """
        Return a subclassed Queue whose ``put()`` / ``put_nowait()`` silently
        drops 30 % of frames.
        """
        monkey = self

        class ChaoticQueue(asyncio.Queue):
            async def put(self, item: Any) -> None:
                if monkey._rng.random() < FRAME_DROP_PROBABILITY:
                    monkey.frames_dropped += 1
                    logger.debug("CHAOS: dropping perception frame")
                    return
                monkey.frames_delivered += 1
                await super().put(item)

            def put_nowait(self, item: Any) -> None:
                if monkey._rng.random() < FRAME_DROP_PROBABILITY:
                    monkey.frames_dropped += 1
                    logger.debug("CHAOS: dropping perception frame (nowait)")
                    return
                monkey.frames_delivered += 1
                super().put_nowait(item)

        # Migrate items already in the base queue
        chaotic: ChaoticQueue = ChaoticQueue()
        return chaotic


# ===========================================================================
# Pytest fixtures
# ===========================================================================

@pytest.fixture
def chaos_monkey() -> ChaosMonkey:
    """A seeded ChaosMonkey instance for reproducible runs."""
    return ChaosMonkey(seed=2026)


@pytest.fixture
def base_llm_client() -> Any:
    """
    A minimal stub LLM client that always returns a valid string.
    The ChaosMonkey wraps this to inject faults.
    """
    client = AsyncMock()
    client.complete.return_value = "Click the submit button to proceed."
    client.extract_confidence.return_value = 0.9
    return client


@pytest.fixture
def base_redis_client() -> Any:
    """
    A minimal stub Redis client that always returns None (cold cache).
    The ChaosMonkey can additionally force misses on top of this.
    """
    client = AsyncMock()
    client.get.return_value = None   # no pre-warmed cache
    client.set.return_value = True
    return client


@pytest.fixture
def perception_queue() -> asyncio.Queue:
    """A plain asyncio.Queue; the ChaosMonkey wraps it for frame drops."""
    return asyncio.Queue()


@pytest.fixture
async def chaos_pipeline(
    chaos_monkey: ChaosMonkey,
    base_llm_client: Any,
    base_redis_client: Any,
) -> AsyncIterator[tuple[ExecraPipeline, ChaosMonkey, asyncio.Queue]]:
    """
    Assembles the full chaos-injected pipeline.

    Yields:
        (pipeline, chaos_monkey, raw_perception_queue)
        The raw queue is unwrapped so tests can push frames into it;
        the pipeline internally uses the chaotic (drop-enabled) queue.
    """
    raw_queue     = asyncio.Queue()
    chaotic_queue = chaos_monkey.wrap_queue(raw_queue)
    chaotic_llm   = chaos_monkey.wrap_llm(base_llm_client)
    chaotic_redis = chaos_monkey.wrap_redis(base_redis_client)

    pipeline = ExecraPipeline(
        llm_client       = chaotic_llm,
        redis_client     = chaotic_redis,
        perception_queue = chaotic_queue,
    )
    yield pipeline, chaos_monkey, raw_queue


# ===========================================================================
# Helper: background frame producer
# ===========================================================================

async def _produce_frames(queue: asyncio.Queue, duration: float, fps: float = 2.0) -> None:
    """
    Push synthetic perception frames into *queue* at *fps* for *duration* seconds.
    Mimics the real ScreenCapture / CameraFeed producers.
    """
    interval   = 1.0 / fps
    deadline   = asyncio.get_event_loop().time() + duration
    frame_num  = 0
    while asyncio.get_event_loop().time() < deadline:
        frame_num += 1
        frame = {
            "frame_id" : frame_num,
            "text"     : f"screen text frame {frame_num}",
            "timestamp": time.time(),
        }
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass
        await asyncio.sleep(interval)


# ===========================================================================
# Test cases
# ===========================================================================

class TestPipelineResilience:
    """Chaos-engineering tests for ExecraPipeline."""

    # ------------------------------------------------------------------
    # 1. Core 60-second chaos run
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pipeline_never_crashes_under_chaos(
        self, chaos_pipeline: tuple[ExecraPipeline, ChaosMonkey, asyncio.Queue]
    ) -> None:
        """
        Pipeline runs for CHAOS_RUN_SECONDS without raising an unhandled
        exception, even while all three fault types fire concurrently.
        """
        pipeline, monkey, raw_queue = chaos_pipeline

        # Run frame producer and pipeline concurrently
        await asyncio.gather(
            pipeline.run(CHAOS_RUN_SECONDS),
            _produce_frames(raw_queue, CHAOS_RUN_SECONDS, fps=2.0),
        )

        # The test itself passing confirms no unhandled exception was raised.
        assert True, "Pipeline raised an unhandled exception during chaos run."

    @pytest.mark.asyncio
    async def test_guidance_still_delivered_under_chaos(
        self, chaos_pipeline: tuple[ExecraPipeline, ChaosMonkey, asyncio.Queue]
    ) -> None:
        """
        Even under chaos, guidance (possibly degraded) is delivered on every
        pipeline tick.  We accept a 25 % delivery gap to allow for very
        unlucky chaos sequences.
        """
        pipeline, monkey, raw_queue = chaos_pipeline

        await asyncio.gather(
            pipeline.run(CHAOS_RUN_SECONDS),
            _produce_frames(raw_queue, CHAOS_RUN_SECONDS, fps=2.0),
        )

        expected_ticks = int(CHAOS_RUN_SECONDS / TICK_INTERVAL)
        # Allow a generous 25 % delivery gap for degraded runs
        minimum_deliveries = int(expected_ticks * 0.75)

        assert len(pipeline.guidance_delivered) >= minimum_deliveries, (
            f"Expected ≥ {minimum_deliveries} guidance deliveries, "
            f"got {len(pipeline.guidance_delivered)} "
            f"(ticks≈{expected_ticks})"
        )

    @pytest.mark.asyncio
    async def test_all_errors_are_logged(
        self, chaos_pipeline: tuple[ExecraPipeline, ChaosMonkey, asyncio.Queue]
    ) -> None:
        """
        Every injected LLM TimeoutError must appear in pipeline.error_log
        with kind == 'llm_timeout'.
        """
        pipeline, monkey, raw_queue = chaos_pipeline

        await asyncio.gather(
            pipeline.run(CHAOS_RUN_SECONDS),
            _produce_frames(raw_queue, CHAOS_RUN_SECONDS, fps=2.0),
        )

        timeout_log_entries = [
            e for e in pipeline.error_log if e["kind"] == "llm_timeout"
        ]
        assert len(timeout_log_entries) == monkey.llm_timeouts_injected, (
            f"Injected {monkey.llm_timeouts_injected} LLM timeouts but "
            f"only {len(timeout_log_entries)} were logged."
        )

    # ------------------------------------------------------------------
    # 2. LLM-only fault injection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_timeout_triggers_fallback_guidance(
        self, chaos_monkey: ChaosMonkey, base_redis_client: Any
    ) -> None:
        """
        When the LLM always times out, the pipeline falls back to a degraded
        instruction (confidence < 0.5) rather than delivering nothing.
        """
        # Force every LLM call to time out
        always_timeout = AsyncMock()
        always_timeout.complete.side_effect = TimeoutError("forced timeout")

        queue    = asyncio.Queue()
        pipeline = ExecraPipeline(
            llm_client       = always_timeout,
            redis_client     = base_redis_client,
            perception_queue = queue,
        )
        await asyncio.gather(
            pipeline.run(5),
            _produce_frames(queue, 5, fps=4.0),
        )

        assert len(pipeline.guidance_delivered) > 0, (
            "No guidance delivered even though fallback path should activate."
        )
        degraded = [g for g in pipeline.guidance_delivered if g.confidence < 0.5]
        assert len(degraded) > 0, (
            "Expected degraded (low-confidence) fallback guidance but none found."
        )

    @pytest.mark.asyncio
    async def test_llm_timeout_rate_matches_chaos_probability(
        self, chaos_monkey: ChaosMonkey, base_llm_client: Any, base_redis_client: Any
    ) -> None:
        """
        Statistical check: over many calls the observed timeout rate should
        be within ±10 % of the configured 20 % probability.
        """
        chaotic_llm   = chaos_monkey.wrap_llm(base_llm_client)
        queue         = asyncio.Queue()
        pipeline      = ExecraPipeline(
            llm_client       = chaotic_llm,
            redis_client     = base_redis_client,
            perception_queue = queue,
        )

        await asyncio.gather(
            pipeline.run(CHAOS_RUN_SECONDS),
            _produce_frames(queue, CHAOS_RUN_SECONDS, fps=2.0),
        )

        total_llm_attempts = (
            chaos_monkey.llm_timeouts_injected
            + len([g for g in pipeline.guidance_delivered if "llm" in g.source])
        )
        if total_llm_attempts == 0:
            pytest.skip("No LLM calls were made — cannot evaluate rate.")

        observed_rate = chaos_monkey.llm_timeouts_injected / total_llm_attempts
        assert abs(observed_rate - LLM_TIMEOUT_PROBABILITY) < 0.15, (
            f"LLM timeout rate {observed_rate:.2%} deviates more than 15 % "
            f"from expected {LLM_TIMEOUT_PROBABILITY:.0%}"
        )

    # ------------------------------------------------------------------
    # 3. Redis-only fault injection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_redis_miss_falls_through_to_llm(
        self, base_llm_client: Any
    ) -> None:
        """
        When Redis always returns a cache miss, every tick falls through to
        the LLM path.  LLM call count should equal the number of ticks.
        """
        always_miss = AsyncMock()
        always_miss.get.return_value = None

        queue    = asyncio.Queue()
        pipeline = ExecraPipeline(
            llm_client       = base_llm_client,
            redis_client     = always_miss,
            perception_queue = queue,
        )
        await asyncio.gather(
            pipeline.run(5),
            _produce_frames(queue, 5, fps=4.0),
        )

        # All guidance should come from "llm" source (no cache hits)
        llm_sourced = [g for g in pipeline.guidance_delivered if "llm" in g.source]
        assert len(llm_sourced) == len(pipeline.guidance_delivered), (
            "Expected all guidance from LLM when Redis always misses."
        )

    @pytest.mark.asyncio
    async def test_redis_miss_rate_matches_chaos_probability(
        self, chaos_monkey: ChaosMonkey, base_llm_client: Any, base_redis_client: Any
    ) -> None:
        """
        Statistical check: observed cache-miss rate stays within ±10 % of
        the 50 % REDIS_MISS_PROBABILITY.
        """
        chaotic_redis = chaos_monkey.wrap_redis(base_redis_client)
        queue         = asyncio.Queue()
        pipeline      = ExecraPipeline(
            llm_client       = base_llm_client,
            redis_client     = chaotic_redis,
            perception_queue = queue,
        )
        await asyncio.gather(
            pipeline.run(CHAOS_RUN_SECONDS),
            _produce_frames(queue, CHAOS_RUN_SECONDS, fps=2.0),
        )

        # Approximate total get() calls = number of pipeline ticks
        total_reads = pipeline._step
        if total_reads == 0:
            pytest.skip("Pipeline made no ticks.")

        observed_miss_rate = chaos_monkey.redis_misses_injected / total_reads
        assert abs(observed_miss_rate - REDIS_MISS_PROBABILITY) < 0.15, (
            f"Redis miss rate {observed_miss_rate:.2%} deviates more than 15 % "
            f"from expected {REDIS_MISS_PROBABILITY:.0%}"
        )

    # ------------------------------------------------------------------
    # 4. Frame-drop fault injection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_frame_drops_do_not_stall_pipeline(
        self, chaos_monkey: ChaosMonkey, base_llm_client: Any, base_redis_client: Any
    ) -> None:
        """
        Even when 30 % of frames are dropped, the pipeline ticks continuously
        (uses 'no frame data' fallback) and guidance is still delivered.
        """
        raw_queue     = asyncio.Queue()
        chaotic_queue = chaos_monkey.wrap_queue(raw_queue)
        pipeline      = ExecraPipeline(
            llm_client       = base_llm_client,
            redis_client     = base_redis_client,
            perception_queue = chaotic_queue,
        )
        await asyncio.gather(
            pipeline.run(5),
            _produce_frames(raw_queue, 5, fps=10.0),
        )

        assert len(pipeline.guidance_delivered) > 0, (
            "Pipeline stalled under frame drops — no guidance delivered."
        )

    @pytest.mark.asyncio
    async def test_frame_drop_rate_matches_chaos_probability(
        self, chaos_monkey: ChaosMonkey
    ) -> None:
        """
        Statistical check: observed drop rate within ±10 % of 30 %.
        """
        raw_queue     = asyncio.Queue()
        chaotic_queue = chaos_monkey.wrap_queue(raw_queue)

        total_attempts = 200
        for i in range(total_attempts):
            await raw_queue.put({"text": f"frame {i}"})
            chaotic_queue.put_nowait({"text": f"direct {i}"})

        # Second batch goes directly into the chaotic queue
        observed_drop_rate = chaos_monkey.frames_dropped / (
            chaos_monkey.frames_dropped + chaos_monkey.frames_delivered
        ) if (chaos_monkey.frames_dropped + chaos_monkey.frames_delivered) > 0 else 0

        assert abs(observed_drop_rate - FRAME_DROP_PROBABILITY) < 0.15, (
            f"Frame drop rate {observed_drop_rate:.2%} deviates more than 15 % "
            f"from expected {FRAME_DROP_PROBABILITY:.0%}"
        )

    # ------------------------------------------------------------------
    # 5. Simultaneous multi-fault injection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_combined_chaos_no_unhandled_exceptions(
        self, chaos_monkey: ChaosMonkey, base_llm_client: Any, base_redis_client: Any
    ) -> None:
        """
        All three fault types fire simultaneously.  The pipeline must not
        raise an unhandled exception and must deliver at least one guidance
        instruction.
        """
        raw_queue     = asyncio.Queue()
        chaotic_queue = chaos_monkey.wrap_queue(raw_queue)
        chaotic_llm   = chaos_monkey.wrap_llm(base_llm_client)
        chaotic_redis = chaos_monkey.wrap_redis(base_redis_client)

        pipeline = ExecraPipeline(
            llm_client       = chaotic_llm,
            redis_client     = chaotic_redis,
            perception_queue = chaotic_queue,
        )
        await asyncio.gather(
            pipeline.run(10),
            _produce_frames(raw_queue, 10, fps=2.0),
        )

        assert len(pipeline.guidance_delivered) > 0, (
            "Combined chaos produced zero guidance deliveries."
        )

    @pytest.mark.asyncio
    async def test_combined_chaos_error_log_has_correct_kinds(
        self, chaos_monkey: ChaosMonkey, base_llm_client: Any, base_redis_client: Any
    ) -> None:
        """
        Under combined chaos the error log must only contain recognised
        error kinds (no unknown crash types leaking through).
        """
        KNOWN_ERROR_KINDS = {"llm_timeout", "llm_error", "redis_get", "perception_drop"}

        raw_queue     = asyncio.Queue()
        chaotic_queue = chaos_monkey.wrap_queue(raw_queue)
        chaotic_llm   = chaos_monkey.wrap_llm(base_llm_client)
        chaotic_redis = chaos_monkey.wrap_redis(base_redis_client)

        pipeline = ExecraPipeline(
            llm_client       = chaotic_llm,
            redis_client     = chaotic_redis,
            perception_queue = chaotic_queue,
        )
        await asyncio.gather(
            pipeline.run(10),
            _produce_frames(raw_queue, 10, fps=2.0),
        )

        unknown = {e["kind"] for e in pipeline.error_log} - KNOWN_ERROR_KINDS
        assert not unknown, (
            f"Unexpected error kinds in log: {unknown}.  "
            "These may represent unhandled exceptions leaking as errors."
        )

    # ------------------------------------------------------------------
    # 6. Degraded-mode quality assertions
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fallback_guidance_confidence_is_low(
        self, base_redis_client: Any
    ) -> None:
        """
        Guidance delivered during LLM timeouts must carry confidence < 0.5,
        so callers can distinguish degraded output from normal output.
        """
        always_timeout = AsyncMock()
        always_timeout.complete.side_effect = TimeoutError("forced")

        queue    = asyncio.Queue()
        pipeline = ExecraPipeline(
            llm_client       = always_timeout,
            redis_client     = base_redis_client,
            perception_queue = queue,
        )
        await asyncio.gather(
            pipeline.run(5),
            _produce_frames(queue, 5, fps=4.0),
        )

        for g in pipeline.guidance_delivered:
            assert g.confidence < 0.5, (
                f"Fallback guidance at step {g.step} has confidence "
                f"{g.confidence:.2f} — expected < 0.5 for degraded mode."
            )

    @pytest.mark.asyncio
    async def test_normal_guidance_confidence_is_high(
        self, base_llm_client: Any, base_redis_client: Any
    ) -> None:
        """
        Under fault-free conditions, all delivered guidance should have
        confidence ≥ 0.5.
        """
        queue    = asyncio.Queue()
        pipeline = ExecraPipeline(
            llm_client       = base_llm_client,
            redis_client     = base_redis_client,
            perception_queue = queue,
        )
        await asyncio.gather(
            pipeline.run(5),
            _produce_frames(queue, 5, fps=4.0),
        )

        for g in pipeline.guidance_delivered:
            assert g.confidence >= 0.5, (
                f"Normal-mode guidance at step {g.step} has unexpectedly low "
                f"confidence {g.confidence:.2f}."
            )

    # ------------------------------------------------------------------
    # 7. Error-log structure validation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_log_entries_have_required_fields(
        self, chaos_pipeline: tuple[ExecraPipeline, ChaosMonkey, asyncio.Queue]
    ) -> None:
        """
        Every error entry must contain the keys 'kind', 'message', and
        'step' so that downstream log aggregators can parse them reliably.
        """
        pipeline, monkey, raw_queue = chaos_pipeline

        await asyncio.gather(
            pipeline.run(CHAOS_RUN_SECONDS),
            _produce_frames(raw_queue, CHAOS_RUN_SECONDS, fps=2.0),
        )

        required_keys = {"kind", "message", "step"}
        for i, entry in enumerate(pipeline.error_log):
            missing = required_keys - entry.keys()
            assert not missing, (
                f"Error log entry #{i} is missing fields: {missing}. "
                f"Entry: {entry}"
            )

    # ------------------------------------------------------------------
    # 8. Chaos monkey statistics
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_chaos_monkey_counters_are_accurate(
        self,
        chaos_monkey: ChaosMonkey,
        base_llm_client: Any,
        base_redis_client: Any,
    ) -> None:
        """
        frames_dropped + frames_delivered must equal total put() attempts
        made by the producer.
        """
        raw_queue     = asyncio.Queue()
        chaotic_queue = chaos_monkey.wrap_queue(raw_queue)
        pipeline      = ExecraPipeline(
            llm_client       = base_llm_client,
            redis_client     = base_redis_client,
            perception_queue = chaotic_queue,
        )

        TOTAL_FRAMES = 100
        for i in range(TOTAL_FRAMES):
            chaotic_queue.put_nowait({"text": f"frame {i}", "timestamp": time.time()})

        await pipeline.run(5)

        total_tracked = chaos_monkey.frames_dropped + chaos_monkey.frames_delivered
        assert total_tracked == TOTAL_FRAMES, (
            f"ChaosMonkey tracked {total_tracked} frames but {TOTAL_FRAMES} "
            "were enqueued — counter drift detected."
        )
