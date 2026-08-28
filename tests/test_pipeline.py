"""
Integration tests for FoodSpoilagePipeline.

Both sub-predictors (YOLOPredictor and ResNetPredictor) are mocked so the
tests verify pipeline orchestration logic without loading real models.

Run:
    pytest tests/test_pipeline.py -v
"""

import sys
import logging
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make both project root and src/ importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_ONE_YOLO_DETECTION = [
    {
        "bbox": [50, 50, 200, 200],
        "class_name": "apple",
        "class_id": 0,
        "confidence": 0.90,
    }
]

_TWO_YOLO_DETECTIONS = [
    {"bbox": [50, 50, 200, 200], "class_name": "apple", "class_id": 0, "confidence": 0.90},
    {"bbox": [210, 50, 350, 200], "class_name": "banana", "class_id": 1, "confidence": 0.80},
]

_RESNET_FRESH = {
    "class": "fresh",
    "class_id": 0,
    "confidence": 0.95,
    "probabilities": {"fresh": 0.95, "spoiled": 0.05},
    "freshness_percentage": 95.0,
}

_RESNET_SPOILED = {
    "class": "spoiled",
    "class_id": 1,
    "confidence": 0.88,
    "probabilities": {"fresh": 0.12, "spoiled": 0.88},
    "freshness_percentage": 12.0,
}


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def _build_pipeline(tmp_path, yolo_detections, resnet_result):
    """
    Create a FoodSpoilagePipeline whose sub-predictors are mocked.

    The patch is only active during __init__ (which is all that is
    needed — __init__ stores the predictor instances on self).
    """
    from src.pipeline.inference import FoodSpoilagePipeline

    # Non-empty placeholder files satisfy file-existence checks inside the
    # real YOLOPredictor / ResNetPredictor constructors (the checks run
    # before the classes are patched out, so we need real files).
    fake_yolo_pt = tmp_path / "yolo.pt"
    fake_resnet_pt = tmp_path / "resnet.pt"
    fake_yolo_pt.write_bytes(b"x" * 64)
    fake_resnet_pt.write_bytes(b"x" * 64)

    mock_yolo = MagicMock()
    mock_yolo.predict.return_value = yolo_detections

    mock_resnet = MagicMock()
    mock_resnet.predict_from_crop.return_value = resnet_result

    with (
        patch("src.pipeline.inference.YOLOPredictor", return_value=mock_yolo),
        patch("src.pipeline.inference.ResNetPredictor", return_value=mock_resnet),
        patch("src.pipeline.inference.setup_logger",
              return_value=logging.getLogger("test_pipeline")),
    ):
        pipeline = FoodSpoilagePipeline(
            yolo_model_path=str(fake_yolo_pt),
            resnet_model_path=str(fake_resnet_pt),
            device="cpu",
        )
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_array():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests — output structure
# ---------------------------------------------------------------------------

class TestPipelineOutputStructure:
    def test_result_has_detections_key(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        result = pipeline.process_image(image_array)
        assert "detections" in result

    def test_result_has_summary_key(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        result = pipeline.process_image(image_array)
        assert "summary" in result

    def test_result_has_metadata_key(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        result = pipeline.process_image(image_array)
        assert "metadata" in result

    def test_summary_contains_counts(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        summary = pipeline.process_image(image_array)["summary"]
        for key in ("total_detections", "fresh_count", "spoiled_count"):
            assert key in summary, f"Missing summary key: {key}"

    def test_detection_entry_contains_required_fields(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        detections = pipeline.process_image(image_array)["detections"]
        assert len(detections) == 1
        det = detections[0]
        for field in (
            "object_class",
            "spoilage_status",
            "spoilage_confidence",
            "object_confidence",
            "combined_confidence",
        ):
            assert field in det, f"Missing detection field: {field}"


# ---------------------------------------------------------------------------
# Tests — counting logic
# ---------------------------------------------------------------------------

class TestPipelineCounting:
    def test_one_fresh_detection(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        summary = pipeline.process_image(image_array)["summary"]
        assert summary["fresh_count"] == 1
        assert summary["spoiled_count"] == 0

    def test_one_spoiled_detection(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_SPOILED)
        summary = pipeline.process_image(image_array)["summary"]
        assert summary["spoiled_count"] == 1
        assert summary["fresh_count"] == 0

    def test_two_detections_total(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _TWO_YOLO_DETECTIONS, _RESNET_FRESH)
        summary = pipeline.process_image(image_array)["summary"]
        assert summary["total_detections"] == 2

    def test_spoilage_status_matches_resnet_output(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_SPOILED)
        detections = pipeline.process_image(image_array)["detections"]
        assert detections[0]["spoilage_status"] == "spoiled"

    def test_object_class_matches_yolo_output(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        detections = pipeline.process_image(image_array)["detections"]
        assert detections[0]["object_class"] == "apple"


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    def test_no_detections_returns_empty_list(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, [], _RESNET_FRESH)
        result = pipeline.process_image(image_array)
        assert result["detections"] == []
        assert result["summary"]["total_detections"] == 0

    def test_pil_image_input_accepted(self, tmp_path):
        pil_img = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), "RGB"
        )
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        result = pipeline.process_image(pil_img)
        assert "detections" in result

    def test_nonexistent_image_path_raises(self, tmp_path):
        pipeline = _build_pipeline(tmp_path, [], _RESNET_FRESH)
        with pytest.raises(FileNotFoundError):
            pipeline.process_image(str(tmp_path / "no_such_image.jpg"))

    def test_metadata_processing_time_is_positive(self, tmp_path, image_array):
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        metadata = pipeline.process_image(image_array)["metadata"]
        assert metadata["processing_time"] >= 0

    def test_combined_confidence_is_average(self, tmp_path, image_array):
        """combined_confidence should be the mean of YOLO and ResNet scores."""
        pipeline = _build_pipeline(tmp_path, _ONE_YOLO_DETECTION, _RESNET_FRESH)
        det = pipeline.process_image(image_array)["detections"][0]
        expected = (det["object_confidence"] + det["spoilage_confidence"]) / 2
        assert abs(det["combined_confidence"] - expected) < 1e-6
