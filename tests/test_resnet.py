"""
Unit tests for the ResNet model architecture and predictor.

All tests use untrained weights or lightweight mocks — no GPU or trained
checkpoint is required to run the suite.

Run:
    pytest tests/test_resnet.py -v
"""

import sys
import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from unittest.mock import patch

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resnet.model import (
    FoodSpoilageResNet,
    build_resnet,
    SUPPORTED_ARCHITECTURES,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def untrained_model():
    """Lightweight untrained resnet50 (no pretrained download)."""
    return build_resnet(architecture="resnet50", num_classes=2, pretrained=False)


@pytest.fixture
def dummy_pil_image():
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def dummy_crop():
    """Numpy crop array in RGB (H, W, C)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_predictor(tmp_path, untrained_model):
    """
    ResNetPredictor backed by a real (untrained) model but with file-system
    checkpoint loading mocked so no trained .pt file is required.
    """
    from resnet.predict_resnet import ResNetPredictor

    fake_pt = tmp_path / "model.pt"
    torch.save(untrained_model.state_dict(), str(fake_pt))

    with patch("resnet.predict_resnet.load_model", return_value=untrained_model):
        predictor = ResNetPredictor(
            model_path=str(fake_pt),
            device="cpu",
        )
    return predictor


# ---------------------------------------------------------------------------
# FoodSpoilageResNet — architecture tests
# ---------------------------------------------------------------------------

class TestFoodSpoilageResNet:
    def test_all_supported_architectures_build(self):
        """Every documented architecture variant should construct without error."""
        for arch in SUPPORTED_ARCHITECTURES:
            m = FoodSpoilageResNet(architecture=arch, pretrained=False)
            assert m is not None, f"{arch} failed to build"

    def test_unsupported_architecture_raises(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            FoodSpoilageResNet(architecture="resnet9999", pretrained=False)

    def test_output_shape_single_sample(self, untrained_model):
        x = torch.randn(1, 3, 224, 224)
        out = untrained_model(x)
        assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"

    def test_output_shape_batch(self, untrained_model):
        x = torch.randn(8, 3, 224, 224)
        out = untrained_model(x)
        assert out.shape == (8, 2)

    def test_custom_num_classes(self):
        m = FoodSpoilageResNet(architecture="resnet18", num_classes=5, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        assert m(x).shape == (1, 5)

    def test_output_is_logits_not_probabilities(self, untrained_model):
        """Raw model output should NOT sum to 1 (it's logits, not softmax)."""
        x = torch.randn(4, 3, 224, 224)
        out = untrained_model(x)
        row_sums = out.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones(4), atol=0.01)


class TestBuildResnet:
    def test_build_default_returns_food_spoilage_resnet(self):
        m = build_resnet(pretrained=False)
        assert isinstance(m, FoodSpoilageResNet)

    def test_build_custom_num_classes(self):
        m = build_resnet(num_classes=3, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        assert m(x).shape == (1, 3)


# ---------------------------------------------------------------------------
# ResNetPredictor — prediction tests
# ---------------------------------------------------------------------------

class TestResNetPredictorOutput:
    def test_predict_returns_required_keys(self, mock_predictor, dummy_pil_image):
        result = mock_predictor.predict(dummy_pil_image)
        for key in ("class", "confidence", "probabilities", "freshness_percentage"):
            assert key in result, f"Missing key: {key}"

    def test_predict_class_is_valid_label(self, mock_predictor, dummy_pil_image):
        result = mock_predictor.predict(dummy_pil_image)
        assert result["class"] in ("fresh", "spoiled")

    def test_predict_confidence_in_range(self, mock_predictor, dummy_pil_image):
        result = mock_predictor.predict(dummy_pil_image)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, mock_predictor, dummy_pil_image):
        result = mock_predictor.predict(dummy_pil_image)
        probs = result["probabilities"]
        assert abs(probs["fresh"] + probs["spoiled"] - 1.0) < 1e-5

    def test_predict_freshness_percentage_range(self, mock_predictor, dummy_pil_image):
        result = mock_predictor.predict(dummy_pil_image)
        assert 0.0 <= result["freshness_percentage"] <= 100.0

    def test_predict_from_crop_numpy_array(self, mock_predictor, dummy_crop):
        result = mock_predictor.predict_from_crop(dummy_crop)
        assert result["class"] in ("fresh", "spoiled")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch_returns_correct_count(self, mock_predictor, dummy_pil_image):
        results = mock_predictor.predict_batch([dummy_pil_image] * 3)
        assert len(results) == 3

    def test_predict_batch_each_has_class(self, mock_predictor, dummy_pil_image):
        results = mock_predictor.predict_batch([dummy_pil_image, dummy_pil_image])
        for r in results:
            assert r["class"] in ("fresh", "spoiled")


class TestResNetPredictorErrors:
    def test_missing_model_file_raises(self, tmp_path):
        from resnet.predict_resnet import ResNetPredictor
        with pytest.raises(FileNotFoundError):
            ResNetPredictor(model_path=str(tmp_path / "nonexistent.pt"))

    def test_missing_image_file_raises(self, mock_predictor, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            mock_predictor.predict(str(tmp_path / "no_image.jpg"))
