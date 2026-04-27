"""Tests for the visual grounding module."""

from __future__ import annotations

import pytest
from PIL import Image

from vision_desktop_automation.grounding import Detection, VisualGrounder, _compute_iou
from vision_desktop_automation.config import Config


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_center_calculation(self) -> None:
        det = Detection(bbox=(100, 200, 300, 400), confidence=0.9, label="test")
        assert det.center == (200, 300)

    def test_center_calculation_small(self) -> None:
        det = Detection(bbox=(10, 10, 20, 20), confidence=0.8, label="icon")
        assert det.center == (15, 15)

    def test_area(self) -> None:
        det = Detection(bbox=(0, 0, 100, 50), confidence=0.7, label="box")
        assert det.area == 5000

    def test_zero_area(self) -> None:
        det = Detection(bbox=(10, 10, 10, 20), confidence=0.5, label="line")
        assert det.area == 0


class TestIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self) -> None:
        box = (0, 0, 100, 100)
        assert _compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 200, 200)
        assert _compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        assert _compute_iou(box1, box2) == pytest.approx(2500 / 17500)

    def test_contained_box(self) -> None:
        outer = (0, 0, 200, 200)
        inner = (50, 50, 100, 100)
        # Intersection: 50x50 = 2500
        # Union: 40000 + 2500 - 2500 = 40000
        assert _compute_iou(outer, inner) == pytest.approx(2500 / 40000)

    def test_zero_area_boxes(self) -> None:
        box1 = (0, 0, 0, 0)
        box2 = (0, 0, 0, 0)
        assert _compute_iou(box1, box2) == 0.0


class TestNMS:
    """Tests for non-maximum suppression."""

    def test_nms_empty(self) -> None:
        assert VisualGrounder._nms([]) == []

    def test_nms_single(self) -> None:
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.9, label="test")
        result = VisualGrounder._nms([det])
        assert len(result) == 1
        assert result[0] is det

    def test_nms_removes_overlapping(self) -> None:
        d1 = Detection(bbox=(0, 0, 100, 100), confidence=0.9, label="high")
        d2 = Detection(bbox=(10, 10, 110, 110), confidence=0.7, label="low")
        result = VisualGrounder._nms([d1, d2], iou_threshold=0.3)
        assert len(result) == 1
        assert result[0].label == "high"

    def test_nms_keeps_distant(self) -> None:
        d1 = Detection(bbox=(0, 0, 50, 50), confidence=0.9, label="a")
        d2 = Detection(bbox=(200, 200, 250, 250), confidence=0.8, label="b")
        result = VisualGrounder._nms([d1, d2])
        assert len(result) == 2


class TestRegionBias:
    """Tests for region hint biasing."""

    def setup_method(self) -> None:
        self.grounder = VisualGrounder.__new__(VisualGrounder)
        self.grounder.config = Config()

    def test_top_left_bias(self) -> None:
        top_left = Detection(bbox=(10, 10, 60, 60), confidence=0.7, label="tl")
        bottom_right = Detection(bbox=(1800, 1000, 1850, 1050), confidence=0.7, label="br")
        result = self.grounder._apply_region_bias(
            [bottom_right, top_left], "top-left", (1920, 1080)
        )
        assert result[0].label == "tl"

    def test_bottom_right_bias(self) -> None:
        top_left = Detection(bbox=(10, 10, 60, 60), confidence=0.7, label="tl")
        bottom_right = Detection(bbox=(1800, 1000, 1850, 1050), confidence=0.7, label="br")
        result = self.grounder._apply_region_bias(
            [top_left, bottom_right], "bottom-right", (1920, 1080)
        )
        assert result[0].label == "br"
