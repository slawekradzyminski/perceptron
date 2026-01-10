"""Tests for digit_cnn module."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from backend.nn.digit_cnn import (
    DigitPrediction,
    DigitRecognizer,
    SimpleCNN,
    get_device,
)


class TestSimpleCNN:
    """Tests for SimpleCNN model."""

    def test_model_creation(self) -> None:
        """Test model instantiation."""
        model = SimpleCNN()
        assert model is not None

    def test_forward_pass_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        model = SimpleCNN()
        model.eval()

        # Create input tensor: (batch=1, channels=1, height=28, width=28)
        x = torch.randn(1, 1, 28, 28)
        output = model(x)

        assert output.shape == (1, 10)

    def test_forward_pass_batch(self) -> None:
        """Test forward pass with batch of inputs."""
        model = SimpleCNN()
        model.eval()

        # Batch of 4 images
        x = torch.randn(4, 1, 28, 28)
        output = model(x)

        assert output.shape == (4, 10)

    def test_predict_proba(self) -> None:
        """Test predict_proba produces valid probabilities."""
        model = SimpleCNN()
        model.eval()

        x = torch.randn(1, 1, 28, 28)
        probs = model.predict_proba(x)

        # Check shape
        assert probs.shape == (1, 10)

        # Check it sums to 1
        np.testing.assert_almost_equal(probs.sum().item(), 1.0, decimal=5)

        # Check all values are in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_parameter_count(self) -> None:
        """Test model has expected parameter count."""
        model = SimpleCNN()
        param_count = sum(p.numel() for p in model.parameters())

        # Should be small (~8K parameters)
        assert param_count < 20000
        assert param_count > 5000


class TestGetDevice:
    """Tests for device selection."""

    def test_returns_device(self) -> None:
        """Test get_device returns a valid device."""
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)


class TestDigitRecognizer:
    """Tests for DigitRecognizer service."""

    def test_creation(self) -> None:
        """Test recognizer instantiation."""
        recognizer = DigitRecognizer()
        assert recognizer is not None

    def test_lazy_model_loading(self) -> None:
        """Test model is lazily loaded."""
        recognizer = DigitRecognizer()

        # Model should not be loaded yet
        assert recognizer._model is None

        # Access model property to trigger loading
        model = recognizer.model

        # Now it should be loaded
        assert model is not None
        assert recognizer._model is not None

    def test_preprocess_image_from_bytes(self) -> None:
        """Test image preprocessing from bytes."""
        recognizer = DigitRecognizer()

        # Create a simple test image
        img = Image.new("L", (100, 100), color=200)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        tensor = recognizer.preprocess_image(image_bytes)

        # Check shape: (1, 1, 28, 28)
        assert tensor.shape == (1, 1, 28, 28)

        # Check normalized to [0, 1]
        assert tensor.min() >= 0
        assert tensor.max() <= 1

    def test_preprocess_inverts_light_background(self) -> None:
        """Test that light backgrounds are inverted (MNIST-style)."""
        recognizer = DigitRecognizer()

        # Create white image with black center (like written digit)
        img = Image.new("L", (28, 28), color=255)
        # Draw black square in center
        for i in range(10, 18):
            for j in range(10, 18):
                img.putpixel((i, j), 0)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        tensor = recognizer.preprocess_image(image_bytes)

        # After inversion, the center should be bright (high values)
        # and edges should be dark (low values)
        center_mean = tensor[0, 0, 10:18, 10:18].mean().item()
        edge_mean = tensor[0, 0, 0:5, 0:5].mean().item()

        assert center_mean > edge_mean

    def test_recognize_returns_prediction(self) -> None:
        """Test recognize returns DigitPrediction."""
        recognizer = DigitRecognizer()

        # Create test image
        img = Image.new("L", (28, 28), color=128)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        result = recognizer.recognize(image_bytes)

        assert isinstance(result, DigitPrediction)
        assert 0 <= result.prediction <= 9
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == 10

    def test_recognize_probabilities_sum_to_one(self) -> None:
        """Test that probabilities sum to 1."""
        recognizer = DigitRecognizer()

        img = Image.new("L", (28, 28), color=100)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        result = recognizer.recognize(image_bytes)

        total_prob = sum(p for _, p in result.probabilities)
        np.testing.assert_almost_equal(total_prob, 1.0, decimal=5)

    def test_get_info(self) -> None:
        """Test get_info returns model information."""
        recognizer = DigitRecognizer()
        info = recognizer.get_info()

        assert "architecture" in info
        assert "input_size" in info
        assert "param_count" in info
        assert "device" in info

        assert info["architecture"] == "SimpleCNN"
        assert info["input_size"] == "28×28"


class TestDigitPrediction:
    """Tests for DigitPrediction dataclass."""

    def test_creation(self) -> None:
        """Test dataclass creation."""
        pred = DigitPrediction(
            prediction=7,
            confidence=0.95,
            probabilities=[(i, 0.1) for i in range(10)],
        )

        assert pred.prediction == 7
        assert pred.confidence == 0.95
        assert len(pred.probabilities) == 10
