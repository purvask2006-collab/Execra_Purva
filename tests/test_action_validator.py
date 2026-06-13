"""
tests/test_action_validator.py

Unit tests for core.physical.action_validator.
All tests use synthetic detection data — no camera or CV runtime required.

Run with:
    pytest tests/test_action_validator.py -v
"""

from __future__ import annotations

import pytest

from core.physical.action_validator import (
    ActionValidator,
    BoundingBox,
    Detection,
    HandResult,
    ValidationResult,
    _CookingValidator,
    _FormFillingValidator,
    _HardwareValidator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────


def bbox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.4) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def det(label: str, x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.4, confidence=0.95) -> Detection:
    return Detection(label=label, confidence=confidence, bbox=bbox(x_min, y_min, x_max, y_max))


def hand(label="Right", x_min=0.05, y_min=0.1, x_max=0.25, y_max=0.4, confidence=0.92) -> HandResult:
    return HandResult(
        hand_label=label,
        confidence=confidence,
        bbox=bbox(x_min, y_min, x_max, y_max),
    )


# ──────────────────────────────────────────────────────────────────────────────
# BoundingBox unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBoundingBox:
    def test_centre(self):
        b = BoundingBox(0.0, 0.0, 0.4, 0.6)
        assert b.centre == pytest.approx((0.2, 0.3))

    def test_area(self):
        b = BoundingBox(0.0, 0.0, 0.5, 0.5)
        assert b.area == pytest.approx(0.25)

    def test_distance_to_same_box_is_zero(self):
        b = BoundingBox(0.1, 0.1, 0.3, 0.3)
        assert b.distance_to(b) == pytest.approx(0.0)

    def test_distance_to_adjacent_box(self):
        b1 = BoundingBox(0.0, 0.0, 0.2, 0.2)   # centre (0.1, 0.1)
        b2 = BoundingBox(0.6, 0.0, 0.8, 0.2)   # centre (0.7, 0.1)
        assert b1.distance_to(b2) == pytest.approx(0.6, abs=1e-6)

    def test_overlaps_identical_boxes(self):
        b = BoundingBox(0.0, 0.0, 0.5, 0.5)
        assert b.overlaps(b, threshold=0.0)

    def test_no_overlap(self):
        b1 = BoundingBox(0.0, 0.0, 0.3, 0.3)
        b2 = BoundingBox(0.7, 0.7, 1.0, 1.0)
        assert not b1.overlaps(b2)


# ──────────────────────────────────────────────────────────────────────────────
# ActionValidator._classify_step
# ──────────────────────────────────────────────────────────────────────────────


class TestClassifyStep:
    validator = ActionValidator()

    @pytest.mark.parametrize("step", [
        "Chop the onions",
        "Slice the bread thinly",
        "Dice the tomatoes",
        "Mince the garlic",
        "Peel the potatoes",
    ])
    def test_cooking_steps(self, step):
        assert self.validator._classify_step(step) == "cooking"

    @pytest.mark.parametrize("step", [
        "Screw in the bolt",
        "Fasten the panel",
        "Assemble the gearbox",
        "Tighten the M4 screw",
        "Install the heatsink",
    ])
    def test_hardware_steps(self, step):
        assert self.validator._classify_step(step) == "hardware"

    @pytest.mark.parametrize("step", [
        "Fill in the application form",
        "Write your name on line 3",
        "Sign at the bottom of the page",
        "Complete the address field",
        "Enter your date of birth",
    ])
    def test_form_steps(self, step):
        assert self.validator._classify_step(step) == "form"

    def test_unknown_step(self):
        assert self.validator._classify_step("Walk to the kitchen") == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Cooking validator
# ──────────────────────────────────────────────────────────────────────────────


class TestCookingValidator:
    v = _CookingValidator()
    step = "Chop the onions"

    # ── happy path ────────────────────────────────────────────────────────────

    def test_valid_scene(self):
        """Knife near board, far from body → valid."""
        detections = [
            det("knife",          x_min=0.3, y_min=0.4, x_max=0.5, y_max=0.6),
            det("cutting board",  x_min=0.25, y_min=0.35, x_max=0.55, y_max=0.65),
            det("person",         x_min=0.0, y_min=0.0, x_max=0.15, y_max=0.25),
        ]
        result = self.v.validate(detections, [hand()], None)
        assert result.is_valid
        assert result.confidence > 0.5
        assert result.issues == []
        assert result.corrections == []

    # ── knife absent ──────────────────────────────────────────────────────────

    def test_no_knife_detected(self):
        detections = [det("cutting board")]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("knife" in i.lower() for i in result.issues)
        assert len(result.corrections) >= 1

    # ── cutting board absent ──────────────────────────────────────────────────

    def test_no_cutting_board(self):
        detections = [det("knife")]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("cutting board" in i.lower() for i in result.issues)

    # ── knife too far from board ───────────────────────────────────────────────

    def test_knife_far_from_board(self):
        """Knife in top-left, board in bottom-right → distance ~0.85 > threshold."""
        detections = [
            det("knife",         x_min=0.0, y_min=0.0, x_max=0.15, y_max=0.15),
            det("cutting board", x_min=0.8, y_min=0.8, x_max=1.0, y_max=1.0),
        ]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("far" in i.lower() or "cutting board" in i.lower() for i in result.issues)
        assert any("cutting board" in c.lower() for c in result.corrections)

    # ── knife near body ───────────────────────────────────────────────────────

    def test_knife_near_body(self):
        """Knife overlapping person bbox → safety violation."""
        detections = [
            det("knife",         x_min=0.3, y_min=0.3, x_max=0.5, y_max=0.6),
            det("cutting board", x_min=0.29, y_min=0.29, x_max=0.55, y_max=0.65),
            # person bbox nearly identical to knife → distance ≈ 0
            det("person",        x_min=0.3, y_min=0.3, x_max=0.5, y_max=0.6),
        ]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("body" in i.lower() or "dangerously" in i.lower() for i in result.issues)
        assert any("body" in c.lower() or "fingers" in c.lower() for c in result.corrections)

    # ── multiple issues ───────────────────────────────────────────────────────

    def test_knife_far_from_board_and_near_body(self):
        """Both distance rules violated simultaneously → two issues."""
        detections = [
            det("knife",         x_min=0.05, y_min=0.05, x_max=0.20, y_max=0.20),
            det("cutting board", x_min=0.80, y_min=0.80, x_max=1.00, y_max=1.00),
            det("person",        x_min=0.05, y_min=0.05, x_max=0.20, y_max=0.20),
        ]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert len(result.issues) >= 2

    # ── label variations ─────────────────────────────────────────────────────

    def test_chopping_board_label(self):
        """'chopping board' should be treated the same as 'cutting board'."""
        detections = [
            det("knife",          x_min=0.3, y_min=0.4, x_max=0.5, y_max=0.6),
            det("chopping board", x_min=0.25, y_min=0.35, x_max=0.55, y_max=0.65),
        ]
        result = self.v.validate(detections, [], None)
        assert result.is_valid

    def test_cleaver_label(self):
        """'cleaver' should be treated as a knife."""
        detections = [
            det("cleaver",        x_min=0.3, y_min=0.4, x_max=0.5, y_max=0.6),
            det("cutting board",  x_min=0.25, y_min=0.35, x_max=0.55, y_max=0.65),
        ]
        result = self.v.validate(detections, [], None)
        assert result.is_valid


# ──────────────────────────────────────────────────────────────────────────────
# Hardware / assembly validator
# ──────────────────────────────────────────────────────────────────────────────


class TestHardwareValidator:
    v = _HardwareValidator()

    # ── happy path ────────────────────────────────────────────────────────────

    def test_vertical_screwdriver_valid(self):
        """Tall bounding box → vertical orientation → valid."""
        detections = [
            # height = 0.6, width = 0.1 → aspect = 6.0 ≥ threshold
            det("screwdriver", x_min=0.4, y_min=0.1, x_max=0.5, y_max=0.7, confidence=0.95),
        ]
        result = self.v.validate(detections, [hand()], None)
        assert result.is_valid
        assert result.confidence > 0.5

    # ── no tool ───────────────────────────────────────────────────────────────

    def test_no_screwdriver_detected(self):
        result = self.v.validate([], [], None)
        assert not result.is_valid
        assert any("screwdriver" in i.lower() or "tool" in i.lower() for i in result.issues)

    # ── horizontal (wrong) orientation ────────────────────────────────────────

    def test_horizontal_screwdriver_invalid(self):
        """Wide bounding box → horizontal orientation → invalid."""
        detections = [
            # height = 0.1, width = 0.6 → aspect ≈ 0.17 < threshold
            det("screwdriver", x_min=0.1, y_min=0.4, x_max=0.7, y_max=0.5, confidence=0.90),
        ]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("horizontal" in i.lower() for i in result.issues)
        assert any("vertical" in c.lower() for c in result.corrections)

    # ── borderline aspect ratio ───────────────────────────────────────────────

    def test_borderline_orientation(self):
        """Aspect ratio exactly at threshold should pass."""
        threshold = _HardwareValidator.VERTICAL_RATIO_THRESHOLD
        # height / width = threshold → should be valid
        width = 0.2
        height = width * threshold
        detections = [
            det("screwdriver",
                x_min=0.3, y_min=0.2,
                x_max=0.3 + width, y_max=0.2 + height,
                confidence=0.88),
        ]
        result = self.v.validate(detections, [], None)
        # Aspect exactly equals threshold, so is_valid depends on strict vs ≥
        # Our implementation uses "<", so equal should be valid
        assert result.is_valid

    # ── depth-map alignment ───────────────────────────────────────────────────

    def test_depth_alignment_correct(self):
        """Tool and workpiece at similar depth → no depth issue."""
        detections = [
            det("screwdriver", x_min=0.4, y_min=0.1, x_max=0.5, y_max=0.7),
            det("pcb",         x_min=0.35, y_min=0.60, x_max=0.55, y_max=0.80),
        ]
        # Uniform depth map at 0.5 m
        depth_map = [[0.5] * 100 for _ in range(100)]
        result = self.v.validate(detections, [], depth_map)
        assert result.is_valid

    def test_depth_alignment_misaligned(self):
        """Tool much closer than workpiece → depth violation."""
        detections = [
            det("screwdriver", x_min=0.4, y_min=0.1, x_max=0.5, y_max=0.7),
            det("pcb",         x_min=0.35, y_min=0.60, x_max=0.55, y_max=0.80),
        ]
        # Build a depth map where tool region = 0.20 m, workpiece region = 0.90 m
        depth_map = [[0.20] * 100 for _ in range(100)]
        for r in range(60, 80):
            for c in range(35, 55):
                depth_map[r][c] = 0.90
        result = self.v.validate(detections, [], depth_map)
        assert not result.is_valid
        assert any("depth" in i.lower() or "misaligned" in i.lower() for i in result.issues)

    # ── alternative tool labels ───────────────────────────────────────────────

    def test_drill_label_accepted(self):
        detections = [det("drill", x_min=0.4, y_min=0.1, x_max=0.5, y_max=0.8)]
        result = self.v.validate(detections, [], None)
        assert result.is_valid


# ──────────────────────────────────────────────────────────────────────────────
# Form-filling validator
# ──────────────────────────────────────────────────────────────────────────────


class TestFormFillingValidator:
    v = _FormFillingValidator()

    # ── happy path ────────────────────────────────────────────────────────────

    def test_pen_in_hand_on_paper(self):
        """Pen near dominant hand, near paper, far from phone → valid."""
        right_hand = hand("Right", x_min=0.30, y_min=0.40, x_max=0.50, y_max=0.60)
        detections = [
            det("pen",    x_min=0.32, y_min=0.42, x_max=0.48, y_max=0.58),
            det("paper",  x_min=0.20, y_min=0.30, x_max=0.70, y_max=0.80),
        ]
        result = self.v.validate(detections, [right_hand], None)
        assert result.is_valid
        assert result.issues == []

    # ── no pen ────────────────────────────────────────────────────────────────

    def test_no_pen_detected(self):
        detections = [det("paper")]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("pen" in i.lower() or "writing" in i.lower() for i in result.issues)

    # ── no paper ─────────────────────────────────────────────────────────────

    def test_no_paper_detected(self):
        detections = [det("pen")]
        result = self.v.validate(detections, [hand()], None)
        assert not result.is_valid
        assert any("paper" in i.lower() or "document" in i.lower() for i in result.issues)

    # ── pen far from paper ────────────────────────────────────────────────────

    def test_pen_far_from_paper(self):
        """Pen top-left, paper bottom-right → exceeds distance threshold."""
        detections = [
            det("pen",   x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1),
            det("paper", x_min=0.8, y_min=0.8, x_max=1.0, y_max=1.0),
        ]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("far" in i.lower() for i in result.issues)

    # ── pen near phone (wrong surface) ───────────────────────────────────────

    def test_pen_near_phone_invalid(self):
        """Writing on a phone screen instead of paper → flagged."""
        detections = [
            det("pen",    x_min=0.30, y_min=0.40, x_max=0.50, y_max=0.60),
            det("paper",  x_min=0.20, y_min=0.30, x_max=0.70, y_max=0.80),
            det("phone",  x_min=0.32, y_min=0.42, x_max=0.48, y_max=0.58),
        ]
        result = self.v.validate(detections, [], None)
        assert not result.is_valid
        assert any("phone" in i.lower() or "screen" in i.lower() for i in result.issues)
        assert any("paper" in c.lower() or "form" in c.lower() for c in result.corrections)

    # ── wrong hand ────────────────────────────────────────────────────────────

    def test_pen_not_in_dominant_hand(self):
        """Pen is far from the right hand → issue raised."""
        right_hand = hand("Right", x_min=0.70, y_min=0.70, x_max=0.90, y_max=0.90)
        detections = [
            det("pen",   x_min=0.05, y_min=0.05, x_max=0.20, y_max=0.20),
            det("paper", x_min=0.02, y_min=0.02, x_max=0.25, y_max=0.25),
        ]
        result = self.v.validate(detections, [right_hand], None)
        assert not result.is_valid
        assert any("dominant" in i.lower() or "hand" in i.lower() for i in result.issues)

    # ── label variants ────────────────────────────────────────────────────────

    def test_pencil_label_accepted(self):
        right_hand = hand("Right", x_min=0.30, y_min=0.40, x_max=0.50, y_max=0.60)
        detections = [
            det("pencil", x_min=0.32, y_min=0.42, x_max=0.48, y_max=0.58),
            det("document", x_min=0.20, y_min=0.30, x_max=0.70, y_max=0.80),
        ]
        result = self.v.validate(detections, [right_hand], None)
        assert result.is_valid

    def test_marker_label_accepted(self):
        right_hand = hand("Right", x_min=0.30, y_min=0.40, x_max=0.50, y_max=0.60)
        detections = [
            det("marker",   x_min=0.32, y_min=0.42, x_max=0.48, y_max=0.58),
            det("notebook", x_min=0.20, y_min=0.30, x_max=0.70, y_max=0.80),
        ]
        result = self.v.validate(detections, [right_hand], None)
        assert result.is_valid


# ──────────────────────────────────────────────────────────────────────────────
# ActionValidator (top-level dispatch)
# ──────────────────────────────────────────────────────────────────────────────


class TestActionValidator:
    validator = ActionValidator()

    def test_cooking_step_dispatched(self):
        detections = [
            det("knife",         x_min=0.3, y_min=0.4, x_max=0.5, y_max=0.6),
            det("cutting board", x_min=0.25, y_min=0.35, x_max=0.55, y_max=0.65),
        ]
        result = self.validator.validate("Slice the tomatoes", detections, [hand()], None)
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_hardware_step_dispatched(self):
        detections = [det("screwdriver", x_min=0.4, y_min=0.1, x_max=0.5, y_max=0.7)]
        result = self.validator.validate("Drive the bolt in", detections, [], None)
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_form_step_dispatched(self):
        right_hand = hand("Right", x_min=0.30, y_min=0.40, x_max=0.50, y_max=0.60)
        detections = [
            det("pen",   x_min=0.32, y_min=0.42, x_max=0.48, y_max=0.58),
            det("paper", x_min=0.20, y_min=0.30, x_max=0.70, y_max=0.80),
        ]
        result = self.validator.validate("Fill in your name", detections, [right_hand], None)
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_unknown_step_returns_optimistic(self):
        result = self.validator.validate("Walk to the fridge", [], [], None)
        assert result.is_valid
        assert result.confidence == pytest.approx(0.50)
        assert result.issues == []

    def test_invalid_cooking_step_includes_corrections(self):
        # Only a knife, no board → invalid
        result = self.validator.validate("Chop the onion", [det("knife")], [], None)
        assert not result.is_valid
        assert len(result.corrections) >= 1

    def test_invalid_hardware_step_includes_corrections(self):
        # Horizontal screwdriver
        detections = [det("screwdriver", x_min=0.1, y_min=0.4, x_max=0.7, y_max=0.5)]
        result = self.validator.validate("Tighten the screw", detections, [], None)
        assert not result.is_valid
        assert len(result.corrections) >= 1

    def test_invalid_form_step_includes_corrections(self):
        result = self.validator.validate("Sign the form", [], [], None)
        assert not result.is_valid
        assert len(result.corrections) >= 1

    # ── ValidationResult.merge ────────────────────────────────────────────────

    def test_validation_result_merge_both_valid(self):
        r1 = ValidationResult(is_valid=True, confidence=0.9)
        r2 = ValidationResult(is_valid=True, confidence=0.8)
        merged = r1.merge(r2)
        assert merged.is_valid
        assert merged.confidence == pytest.approx(0.8)

    def test_validation_result_merge_one_invalid(self):
        r1 = ValidationResult(is_valid=True, confidence=0.9, issues=[], corrections=[])
        r2 = ValidationResult(is_valid=False, confidence=0.4,
                              issues=["bad"], corrections=["fix it"])
        merged = r1.merge(r2)
        assert not merged.is_valid
        assert merged.confidence == pytest.approx(0.4)
        assert "bad" in merged.issues
        assert "fix it" in merged.corrections

    def test_validation_result_merge_issues_concatenated(self):
        r1 = ValidationResult(is_valid=False, confidence=0.5,
                              issues=["issue A"], corrections=["fix A"])
        r2 = ValidationResult(is_valid=False, confidence=0.3,
                              issues=["issue B"], corrections=["fix B"])
        merged = r1.merge(r2)
        assert merged.issues == ["issue A", "issue B"]
        assert merged.corrections == ["fix A", "fix B"]
