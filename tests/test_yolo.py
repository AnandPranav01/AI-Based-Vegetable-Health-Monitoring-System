"""
Unit tests for the YOLO predictor.

The Ultralytics YOLO model is mocked so no trained weights or GPU are needed.

Run:
    pytest tests/test_yolo.py -v
"""

import sys
import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_boxes_mock(detections):
    """
    Build a mock `result.boxes` object whose attributes behave like
    real Ultralytics tensors.

    detections: list of (xyxy, conf, cls_id) tuples
        e.g. [([10, 20, 100, 120], 0.85, 0)]
    """
    if detections:
        xyxy = torch.tensor(
            [[d[0][0], d[0][1], d[0][2], d[0][3]] for d in detections],
            dtype=torch.float32,
        )
        conf = torch.tensor([d[1] for d in detections], dtype=torch.float32)
        cls = torch.tensor([d[2] for d in detections], dtype=torch.float32)
    else:
        xyxy = torch.zeros((0, 4))
        conf = torch.zeros(0)
        cls = torch.zeros(0)

    boxes = MagicMock()
    boxes.xyxy = xyxy
    boxes.conf = conf
    boxes.cls = cls
    boxes.__len__ = MagicMock(return_value=len(detections))
    return boxes


def _make_yolo_result(detections=None):
    """Build a single fake Ultralytics Results object."""
    if detections is None:
        detections = [([10, 20, 100, 120], 0.9, 0)]
    result = MagicMock()
    result.boxes = _make_boxes_mock(detections)
    result.names = {0: "apple", 1: "banana", 2: "tomato"}
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fake_yolo_model():
    """Mock YOLO model returning one apple detection."""
    model = MagicMock()
    model.names = {0: "apple", 1: "banana", 2: "tomato"}
    model.return_value = [_make_yolo_result()]
    return model


@pytest.fixture
def yolo_predictor(tmp_path, fake_yolo_model):
    """YOLOPredictor with YOLO class patched."""
    from yolo.predict_yolo import YOLOPredictor

    fake_pt = tmp_path / "yolo_best.pt"
    fake_pt.write_bytes(b"placeholder" * 10)

    with patch("yolo.predict_yolo.YOLO", return_value=fake_yolo_model):
        predictor = YOLOPredictor(model_path=str(fake_pt), device="cpu")
    return predictor


# ---------------------------------------------------------------------------
# Tests — basic prediction
# ---------------------------------------------------------------------------

class TestYOLOPredictorBasic:
    def test_predict_returns_list(self, yolo_predictor, dummy_image):
        result = yolo_predictor.predict(dummy_image, return_format="dict")
        assert isinstance(result, list)

    def test_predict_detection_has_required_keys(self, yolo_predictor, dummy_image):
        detections = yolo_predictor.predict(dummy_image, return_format="dict")
        assert len(detections) > 0
        for key in ("bbox", "confidence", "class_id", "class_name"):
            assert key in detections[0], f"Missing key: {key}"

    def test_predict_confidence_in_valid_range(self, yolo_predictor, dummy_image):
        detections = yolo_predictor.predict(dummy_image, return_format="dict")
        for det in detections:
            assert 0.0 <= det["confidence"] <= 1.0

    def test_predict_bbox_has_four_coordinates(self, yolo_predictor, dummy_image):
        detections = yolo_predictor.predict(dummy_image, return_format="dict")
        for det in detections:
            assert len(det["bbox"]) == 4

    def test_predict_bbox_x2_gt_x1(self, yolo_predictor, dummy_image):
        detections = yolo_predictor.predict(dummy_image, return_format="dict")
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            assert x2 > x1, "x2 must be greater than x1"
            assert y2 > y1, "y2 must be greater than y1"

    def test_predict_class_name_is_string(self, yolo_predictor, dummy_image):
        detections = yolo_predictor.predict(dummy_image, return_format="dict")
        for det in detections:
            assert isinstance(det["class_name"], str)


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestYOLOPredictorEdgeCases:
    def test_no_detections_returns_empty_list(self, tmp_path):
        """YOLO returning zero boxes → empty list."""
        from yolo.predict_yolo import YOLOPredictor

        empty_model = MagicMock()
        empty_model.names = {0: "apple"}
        empty_model.return_value = [_make_yolo_result(detections=[])]

        fake_pt = tmp_path / "empty.pt"
        fake_pt.write_bytes(b"placeholder" * 10)

        with patch("yolo.predict_yolo.YOLO", return_value=empty_model):
            predictor = YOLOPredictor(model_path=str(fake_pt), device="cpu")

        result = predictor.predict(
            np.zeros((480, 640, 3), dtype=np.uint8),
            return_format="dict",
        )
        assert result == []

    def test_multiple_detections(self, tmp_path):
        """Multiple detections are all returned."""
        from yolo.predict_yolo import YOLOPredictor

        multi_result = _make_yolo_result(
            detections=[
                ([10, 20, 100, 120], 0.9, 0),
                ([200, 50, 350, 200], 0.75, 1),
                ([400, 10, 500, 90], 0.60, 2),
            ]
        )
        multi_model = MagicMock()
        multi_model.names = {0: "apple", 1: "banana", 2: "tomato"}
        multi_model.return_value = [multi_result]

        fake_pt = tmp_path / "multi.pt"
        fake_pt.write_bytes(b"placeholder" * 10)

        with patch("yolo.predict_yolo.YOLO", return_value=multi_model):
            predictor = YOLOPredictor(model_path=str(fake_pt), device="cpu")

        detections = predictor.predict(
            np.zeros((480, 640, 3), dtype=np.uint8),
            return_format="dict",
        )
        assert len(detections) == 3

    def test_pil_image_input(self, yolo_predictor):
        pil_img = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), "RGB"
        )
        result = yolo_predictor.predict(pil_img, return_format="dict")
        assert isinstance(result, list)

    def test_missing_model_raises_file_not_found(self, tmp_path):
        from yolo.predict_yolo import YOLOPredictor
        with pytest.raises(FileNotFoundError):
            YOLOPredictor(model_path=str(tmp_path / "missing.pt"))
