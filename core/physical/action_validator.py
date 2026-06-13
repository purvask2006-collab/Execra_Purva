"""
core/physical/action_validator.py

Validates that the user performed a physical step correctly and safely
before Execra advances to the next step.

Supports three task domains out of the box:
  - cooking  : knife/cutting-board proximity & body-safety checks
  - hardware : screwdriver orientation check
  - form     : pen hand + paper proximity check

The public surface is intentionally small:
    validator = ActionValidator()
    result    = validator.validate(step, detections, hand_results, depth_map)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BoundingBox:
    """Normalised [0, 1] bounding box: (x_min, y_min, x_max, y_max)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def distance_to(self, other: "BoundingBox") -> float:
        """Euclidean distance between the two centres."""
        cx1, cy1 = self.centre
        cx2, cy2 = other.centre
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def overlaps(self, other: "BoundingBox", threshold: float = 0.0) -> bool:
        """True when the intersection-over-union > *threshold*."""
        inter_x = max(0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        inter_y = max(0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        inter = inter_x * inter_y
        union = self.area + other.area - inter
        if union <= 0:
            return False
        return (inter / union) > threshold


@dataclass
class Detection:
    """A single object detection from a CV model."""

    label: str
    confidence: float
    bbox: BoundingBox
    depth: Optional[float] = None          # metres, if available
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandResult:
    """Pose result for one detected hand."""

    hand_label: str                        # "Left" | "Right"
    confidence: float
    bbox: BoundingBox
    landmarks: Optional[List[Tuple[float, float]]] = None   # 21 points, normalised
    holding: Optional[str] = None         # label of the held object, if resolved


@dataclass
class ValidationResult:
    """Return value of :meth:`ActionValidator.validate`."""

    is_valid: bool
    confidence: float                      # overall confidence [0, 1]
    issues: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)

    # convenience ──────────────────────────────────────────────────────────────

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Combine two results (AND semantics: both must be valid)."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            confidence=min(self.confidence, other.confidence),
            issues=self.issues + other.issues,
            corrections=self.corrections + other.corrections,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Task-domain keyword sets
# ──────────────────────────────────────────────────────────────────────────────

_COOKING_KEYWORDS = re.compile(
    r"\b(cut|slice|chop|dice|mince|peel|carve|trim|halve|quarter)\b", re.IGNORECASE
)
_HARDWARE_KEYWORDS = re.compile(
    r"\b(screw|fasten|assemble|tighten|install|attach|mount|drive)\b", re.IGNORECASE
)
_FORM_KEYWORDS = re.compile(
    r"\b(fill|write|sign|complete|enter|record|note|annotate)\b", re.IGNORECASE
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _find_by_label(detections: List[Detection], *labels: str) -> List[Detection]:
    """Return detections whose label matches any of *labels* (case-insensitive)."""
    lowered = {lbl.lower() for lbl in labels}
    return [d for d in detections if d.label.lower() in lowered]


def _dominant_hand(hand_results: List[HandResult]) -> Optional[HandResult]:
    """
    Return the dominant (right) hand result, falling back to whichever hand
    is available if no right hand is detected.
    """
    rights = [h for h in hand_results if h.hand_label.lower() == "right"]
    if rights:
        return max(rights, key=lambda h: h.confidence)
    if hand_results:
        return max(hand_results, key=lambda h: h.confidence)
    return None


def _average_depth(
    depth_map: Optional[List[List[float]]], bbox: BoundingBox
) -> Optional[float]:
    """Sample the mean depth inside *bbox* from a 2-D depth map (H×W, metres)."""
    if depth_map is None:
        return None
    h = len(depth_map)
    w = len(depth_map[0]) if h else 0
    if h == 0 or w == 0:
        return None
    r0 = int(bbox.y_min * h)
    r1 = max(r0 + 1, int(bbox.y_max * h))
    c0 = int(bbox.x_min * w)
    c1 = max(c0 + 1, int(bbox.x_max * w))
    samples = [
        depth_map[r][c]
        for r in range(r0, min(r1, h))
        for c in range(c0, min(c1, w))
        if depth_map[r][c] is not None
    ]
    return sum(samples) / len(samples) if samples else None


# ──────────────────────────────────────────────────────────────────────────────
# Domain validators
# ──────────────────────────────────────────────────────────────────────────────


class _CookingValidator:
    """
    Validates cutting steps:
      1. A knife must be detected.
      2. The knife must be near a cutting board (proximity ≤ MAX_KNIFE_BOARD_DIST).
      3. The knife must NOT be near the user's torso/body
         (proximity ≥ MIN_KNIFE_BODY_DIST).
    """

    MAX_KNIFE_BOARD_DIST: float = 0.35     # normalised image units
    MIN_KNIFE_BODY_DIST: float = 0.30

    # Labels that map to "torso / body region"
    BODY_LABELS = ("person", "torso", "body", "hand", "wrist")
    BOARD_LABELS = ("cutting board", "chopping board", "cutting_board", "chopping_board")
    KNIFE_LABELS = ("knife", "cleaver", "blade")

    def validate(
        self,
        detections: List[Detection],
        hand_results: List[HandResult],
        depth_map: Optional[List[List[float]]],
    ) -> ValidationResult:
        issues: List[str] = []
        corrections: List[str] = []

        knives = _find_by_label(detections, *self.KNIFE_LABELS)
        boards = _find_by_label(detections, *self.BOARD_LABELS)
        bodies = _find_by_label(detections, *self.BODY_LABELS)

        # ── knife presence ────────────────────────────────────────────────────
        if not knives:
            issues.append("No knife detected in the scene.")
            corrections.append(
                "Please ensure the knife is visible to the camera before continuing."
            )
            return ValidationResult(
                is_valid=False, confidence=0.9, issues=issues, corrections=corrections
            )

        knife = max(knives, key=lambda d: d.confidence)

        # ── cutting board presence ────────────────────────────────────────────
        if not boards:
            issues.append("No cutting board detected.")
            corrections.append(
                "Place the item you are cutting on a cutting board and position "
                "both the board and knife within the camera's view."
            )
            return ValidationResult(
                is_valid=False, confidence=0.85, issues=issues, corrections=corrections
            )

        board = min(boards, key=lambda b: knife.bbox.distance_to(b.bbox))
        knife_board_dist = knife.bbox.distance_to(board.bbox)

        # ── knife near board ──────────────────────────────────────────────────
        if knife_board_dist > self.MAX_KNIFE_BOARD_DIST:
            issues.append(
                f"Knife is too far from the cutting board "
                f"(distance={knife_board_dist:.2f}, threshold={self.MAX_KNIFE_BOARD_DIST})."
            )
            corrections.append(
                "Move the cutting action directly over the cutting board."
            )

        # ── knife not near body ───────────────────────────────────────────────
        if bodies:
            nearest_body = min(bodies, key=lambda b: knife.bbox.distance_to(b.bbox))
            knife_body_dist = knife.bbox.distance_to(nearest_body.bbox)
            if knife_body_dist < self.MIN_KNIFE_BODY_DIST:
                issues.append(
                    f"Knife is dangerously close to the body "
                    f"(distance={knife_body_dist:.2f}, minimum safe={self.MIN_KNIFE_BODY_DIST})."
                )
                corrections.append(
                    "Keep the knife pointed away from your body and fingers. "
                    "Use a stable grip on the food and curl your fingertips inward."
                )

        confidence = knife.confidence * board.confidence if not issues else 0.3
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=min(1.0, confidence),
            issues=issues,
            corrections=corrections,
        )


class _HardwareValidator:
    """
    Validates assembly/screw-driving steps:
      1. A screwdriver must be detected.
      2. The screwdriver must be held roughly vertically
         (the bounding-box aspect ratio is tall, not wide).
      3. Depth-map check: the tip must be at a similar depth to the workpiece.
    """

    # If bbox_height / bbox_width > this ratio, orientation is "vertical"
    VERTICAL_RATIO_THRESHOLD: float = 1.3
    TOOL_LABELS = (
        "screwdriver",
        "drill",
        "wrench",
        "spanner",
        "power tool",
        "power_tool",
    )
    WORKPIECE_LABELS = ("pcb", "circuit board", "device", "component", "workpiece", "object")
    # Maximum depth delta (metres) between tool tip and workpiece
    MAX_DEPTH_DELTA: float = 0.20

    def validate(
        self,
        detections: List[Detection],
        hand_results: List[HandResult],
        depth_map: Optional[List[List[float]]],
    ) -> ValidationResult:
        issues: List[str] = []
        corrections: List[str] = []

        tools = _find_by_label(detections, *self.TOOL_LABELS)

        # ── tool presence ─────────────────────────────────────────────────────
        if not tools:
            issues.append("No screwdriver or assembly tool detected.")
            corrections.append(
                "Hold the screwdriver so that it is clearly visible to the camera."
            )
            return ValidationResult(
                is_valid=False, confidence=0.90, issues=issues, corrections=corrections
            )

        tool = max(tools, key=lambda d: d.confidence)
        bbox = tool.bbox
        height = max(1e-6, bbox.y_max - bbox.y_min)
        width = max(1e-6, bbox.x_max - bbox.x_min)
        aspect = height / width

        # ── orientation check ─────────────────────────────────────────────────
        if aspect < self.VERTICAL_RATIO_THRESHOLD:
            issues.append(
                f"Screwdriver appears to be held horizontally "
                f"(aspect ratio height/width={aspect:.2f}, "
                f"required ≥ {self.VERTICAL_RATIO_THRESHOLD})."
            )
            corrections.append(
                "Rotate the screwdriver so the shaft points straight down "
                "into the screw head. A vertical orientation ensures proper torque "
                "and avoids stripping the screw."
            )

        # ── depth alignment (optional, requires depth_map) ────────────────────
        if depth_map is not None:
            workpieces = _find_by_label(detections, *self.WORKPIECE_LABELS)
            tool_depth = _average_depth(depth_map, bbox)
            if workpieces and tool_depth is not None:
                wp = min(workpieces, key=lambda w: tool.bbox.distance_to(w.bbox))
                wp_depth = _average_depth(depth_map, wp.bbox)
                if wp_depth is not None:
                    delta = abs(tool_depth - wp_depth)
                    if delta > self.MAX_DEPTH_DELTA:
                        issues.append(
                            f"Tool tip depth ({tool_depth:.2f} m) is misaligned with "
                            f"workpiece depth ({wp_depth:.2f} m) — Δ={delta:.2f} m."
                        )
                        corrections.append(
                            "Position the screwdriver tip directly above the screw "
                            "before applying pressure."
                        )

        confidence = tool.confidence if not issues else 0.35
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=min(1.0, confidence),
            issues=issues,
            corrections=corrections,
        )


class _FormFillingValidator:
    """
    Validates form-filling / signing steps:
      1. A pen must be detected in the dominant hand's region.
      2. The pen must be near a paper/document (not near a phone/tablet).
    """

    PEN_LABELS = ("pen", "pencil", "marker", "stylus")
    PAPER_LABELS = ("paper", "document", "form", "notebook", "sheet")
    PHONE_LABELS = ("phone", "smartphone", "mobile", "tablet", "screen", "phone screen")

    MAX_PEN_PAPER_DIST: float = 0.30
    MIN_PEN_PHONE_DIST: float = 0.25

    def validate(
        self,
        detections: List[Detection],
        hand_results: List[HandResult],
        depth_map: Optional[List[List[float]]],
    ) -> ValidationResult:
        issues: List[str] = []
        corrections: List[str] = []

        pens = _find_by_label(detections, *self.PEN_LABELS)
        papers = _find_by_label(detections, *self.PAPER_LABELS)
        phones = _find_by_label(detections, *self.PHONE_LABELS)

        # ── pen presence ──────────────────────────────────────────────────────
        if not pens:
            issues.append("No pen or writing instrument detected.")
            corrections.append(
                "Pick up a pen and hold it so it is visible to the camera."
            )
            return ValidationResult(
                is_valid=False, confidence=0.90, issues=issues, corrections=corrections
            )

        pen = max(pens, key=lambda d: d.confidence)

        # ── pen in dominant hand ──────────────────────────────────────────────
        dominant = _dominant_hand(hand_results)
        if dominant is not None:
            pen_hand_dist = pen.bbox.distance_to(dominant.bbox)
            if pen_hand_dist > 0.15:
                issues.append(
                    f"Pen does not appear to be in the dominant hand "
                    f"(distance to dominant hand bbox={pen_hand_dist:.2f})."
                )
                corrections.append(
                    "Hold the pen in your dominant (writing) hand."
                )

        # ── pen near paper ────────────────────────────────────────────────────
        if not papers:
            issues.append("No paper or document detected near the pen.")
            corrections.append(
                "Place the form or document flat on the desk within the camera's view "
                "before writing."
            )
        else:
            nearest_paper = min(papers, key=lambda p: pen.bbox.distance_to(p.bbox))
            pen_paper_dist = pen.bbox.distance_to(nearest_paper.bbox)
            if pen_paper_dist > self.MAX_PEN_PAPER_DIST:
                issues.append(
                    f"Pen is too far from the paper "
                    f"(distance={pen_paper_dist:.2f}, threshold={self.MAX_PEN_PAPER_DIST})."
                )
                corrections.append(
                    "Move the pen so that it is positioned over the form before writing."
                )

        # ── pen not near phone (digital device) ───────────────────────────────
        if phones:
            nearest_phone = min(phones, key=lambda ph: pen.bbox.distance_to(ph.bbox))
            pen_phone_dist = pen.bbox.distance_to(nearest_phone.bbox)
            if pen_phone_dist < self.MIN_PEN_PHONE_DIST:
                issues.append(
                    f"Pen is near a phone/screen rather than paper "
                    f"(distance={pen_phone_dist:.2f})."
                )
                corrections.append(
                    "Write on the physical paper form, not on a phone or tablet screen."
                )

        confidence = pen.confidence if not issues else 0.30
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=min(1.0, confidence),
            issues=issues,
            corrections=corrections,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


class ActionValidator:
    """
    Entry point for physical-action validation.

    Usage::

        validator = ActionValidator()
        result = validator.validate(
            step="Chop the onions",
            detections=[...],      # List[Detection]
            hand_results=[...],    # List[HandResult]
            depth_map=None,        # Optional 2-D list (H x W), metres
        )
        if not result.is_valid:
            dispatch_guidance(result.corrections)
    """

    def __init__(self) -> None:
        self._cooking = _CookingValidator()
        self._hardware = _HardwareValidator()
        self._form = _FormFillingValidator()

    # ── public ────────────────────────────────────────────────────────────────

    def validate(
        self,
        step: str,
        detections: List[Detection],
        hand_results: List[HandResult],
        depth_map: Optional[List[List[float]]] = None,
    ) -> ValidationResult:
        """
        Validate *step* against the current CV observations.

        Parameters
        ----------
        step:
            Natural-language description of the current task step,
            e.g. ``"Chop the carrots"`` or ``"Drive the M3 screw"``.
        detections:
            Object detections from the current frame.
        hand_results:
            Hand-pose results from the current frame.
        depth_map:
            Optional H×W depth map (metres).  When provided, depth-aware
            checks are enabled.

        Returns
        -------
        ValidationResult
            ``is_valid=True`` if the step was performed correctly.
            ``is_valid=False`` means at least one safety / correctness rule
            failed; ``corrections`` carries actionable guidance for the user.
        """
        domain = self._classify_step(step)

        if domain == "cooking":
            return self._cooking.validate(detections, hand_results, depth_map)
        if domain == "hardware":
            return self._hardware.validate(detections, hand_results, depth_map)
        if domain == "form":
            return self._form.validate(detections, hand_results, depth_map)

        # Unknown domain → optimistic pass, low confidence
        return ValidationResult(
            is_valid=True,
            confidence=0.50,
            issues=[],
            corrections=[],
        )

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_step(step: str) -> str:
        """Map a step description to one of the known task domains."""
        if _COOKING_KEYWORDS.search(step):
            return "cooking"
        if _HARDWARE_KEYWORDS.search(step):
            return "hardware"
        if _FORM_KEYWORDS.search(step):
            return "form"
        return "unknown"
