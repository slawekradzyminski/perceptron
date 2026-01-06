"""Tests for DeepMlp (Chapter 4)."""

import pytest

from backend.nn.deep_mlp import (
    DeepMlp,
    DeepMlpForward,
    DeepMlpState,
    DeepMlpStep,
    RegionInfo,
    cross_entropy_loss,
    relu,
    relu_prime,
    softmax,
)


class TestRelu:
    def test_relu_positive(self) -> None:
        assert relu(5.0) == 5.0
        assert relu(0.1) == 0.1

    def test_relu_negative(self) -> None:
        assert relu(-5.0) == 0.0
        assert relu(-0.1) == 0.0

    def test_relu_zero(self) -> None:
        assert relu(0.0) == 0.0


class TestReluPrime:
    def test_relu_prime_positive(self) -> None:
        assert relu_prime(5.0) == 1.0
        assert relu_prime(0.1) == 1.0

    def test_relu_prime_negative(self) -> None:
        assert relu_prime(-5.0) == 0.0
        assert relu_prime(-0.1) == 0.0

    def test_relu_prime_zero(self) -> None:
        assert relu_prime(0.0) == 0.0


class TestSoftmax:
    def test_softmax_two_classes(self) -> None:
        probs = softmax([0.0, 0.0])
        assert len(probs) == 2
        assert abs(probs[0] - 0.5) < 1e-6
        assert abs(probs[1] - 0.5) < 1e-6
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_softmax_large_difference(self) -> None:
        probs = softmax([10.0, 0.0])
        assert probs[0] > 0.99
        assert probs[1] < 0.01

    def test_softmax_three_classes(self) -> None:
        probs = softmax([1.0, 2.0, 3.0])
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6
        # Ordering should be preserved
        assert probs[0] < probs[1] < probs[2]

    def test_softmax_numerical_stability(self) -> None:
        # Large values should not cause overflow
        probs = softmax([1000.0, 1001.0])
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-6
        assert probs[1] > probs[0]


class TestCrossEntropyLoss:
    def test_perfect_prediction(self) -> None:
        # When prediction is near 1 for correct class, loss should be small
        loss = cross_entropy_loss([0.01, 0.99], target=1)
        assert loss < 0.1

    def test_wrong_prediction(self) -> None:
        # When prediction is low for correct class, loss should be high
        loss = cross_entropy_loss([0.99, 0.01], target=1)
        assert loss > 4.0

    def test_uniform_prediction(self) -> None:
        # Uniform distribution gives log(2) ≈ 0.693 loss
        loss = cross_entropy_loss([0.5, 0.5], target=0)
        assert abs(loss - 0.693) < 0.01


class TestDeepMlp:
    def test_init_single_layer(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2, seed=42)
        assert model.depth == 1
        assert model.width == 8
        assert model.param_count == 2 * 8 + 8 + 8 * 2 + 2  # input->hidden + hidden->output

    def test_init_multi_layer(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[16, 16, 16], output_dim=2, seed=42)
        assert model.depth == 3
        assert model.width == 16
        # 2*16 + 16 + 16*16 + 16 + 16*16 + 16 + 16*2 + 2 = 32+16+256+16+256+16+32+2 = 626
        expected = (2 * 16 + 16) + 2 * (16 * 16 + 16) + (16 * 2 + 2)
        assert model.param_count == expected

    def test_forward_shape(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[8, 8], output_dim=2, seed=42)
        fwd = model.forward([0.5, -0.5])
        assert len(fwd.hidden_z) == 2
        assert len(fwd.hidden_a) == 2
        assert len(fwd.hidden_z[0]) == 8
        assert len(fwd.hidden_a[0]) == 8
        assert len(fwd.output_logits) == 2
        assert len(fwd.output_probs) == 2
        assert abs(sum(fwd.output_probs) - 1.0) < 1e-6

    def test_forward_deterministic(self) -> None:
        model1 = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2, seed=42)
        model2 = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2, seed=42)
        fwd1 = model1.forward([0.5, -0.5])
        fwd2 = model2.forward([0.5, -0.5])
        assert fwd1.output_probs == fwd2.output_probs

    def test_activation_signature(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[4, 4], output_dim=2, seed=42)
        fwd = model.forward([0.5, -0.5])
        # Signature should have 4+4=8 bits
        assert len(fwd.activation_signature) == 8
        assert all(bit in (0, 1) for bit in fwd.activation_signature)

    def test_step_reduces_loss(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2, lr=0.1, seed=42)
        x = [0.5, -0.5]
        y = 1

        # Get initial prediction
        fwd_before = model.forward(x)
        loss_before = -__import__("math").log(max(fwd_before.output_probs[y], 1e-15))

        # Take a step
        step = model.step(x, y)
        assert step.loss == pytest.approx(loss_before, rel=1e-6)

        # Loss should decrease after step
        fwd_after = model.forward(x)
        loss_after = -__import__("math").log(max(fwd_after.output_probs[y], 1e-15))
        assert loss_after < loss_before

    def test_region_counting(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[4], output_dim=2, seed=42)
        regions = model.count_regions(resolution=20)
        assert regions.count >= 1
        assert regions.theoretical_max >= regions.count
        # Should have some signatures
        assert len(regions.signatures) == regions.count

    def test_predict_batch(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[4], output_dim=2, seed=42)
        predictions, region_ids = model.predict_batch(resolution=10)
        assert len(predictions) == 10
        assert len(predictions[0]) == 10
        assert len(region_ids) == 10
        assert len(region_ids[0]) == 10
        # All predictions should be 0 or 1
        for row in predictions:
            assert all(p in (0, 1) for p in row)

    def test_get_state(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[8, 8], output_dim=2, seed=42)
        state = model.get_state()
        assert state.input_dim == 2
        assert state.hidden_dims == [8, 8]
        assert state.output_dim == 2
        assert len(state.hidden_weights) == 2
        assert len(state.hidden_biases) == 2


class TestDeepMlpValidation:
    def test_invalid_input_dim(self) -> None:
        with pytest.raises(ValueError, match="input_dim must be positive"):
            DeepMlp(input_dim=0, hidden_dims=[8], output_dim=2)

    def test_invalid_output_dim(self) -> None:
        with pytest.raises(ValueError, match="output_dim must be positive"):
            DeepMlp(input_dim=2, hidden_dims=[8], output_dim=0)

    def test_invalid_hidden_dim(self) -> None:
        with pytest.raises(ValueError, match="hidden_dims.*must be positive"):
            DeepMlp(input_dim=2, hidden_dims=[8, 0, 4], output_dim=2)

    def test_wrong_input_size(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2)
        with pytest.raises(ValueError, match="Expected input dim"):
            model.forward([0.5])  # Only 1 element instead of 2

    def test_invalid_target(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2)
        with pytest.raises(ValueError, match="Target y must be in"):
            model.step([0.5, 0.5], y=5)  # Target out of range


class TestDeepMlpForwardDataclass:
    def test_prediction_property(self) -> None:
        fwd = DeepMlpForward(
            hidden_z=[[1.0, -1.0]],
            hidden_a=[[1.0, 0.0]],
            output_logits=[0.3, 0.7],
            output_probs=[0.4, 0.6],
            activation_signature=(1, 0),
        )
        assert fwd.prediction == 1  # argmax of [0.4, 0.6]

    def test_prediction_ties(self) -> None:
        fwd = DeepMlpForward(
            hidden_z=[],
            hidden_a=[],
            output_logits=[0.5, 0.5],
            output_probs=[0.5, 0.5],
            activation_signature=(),
        )
        # In case of tie, should return first index (0)
        assert fwd.prediction == 0


class TestDeepMlpStepDataclass:
    def test_step_attributes(self) -> None:
        fwd = DeepMlpForward(
            hidden_z=[[0.5]],
            hidden_a=[[0.5]],
            output_logits=[0.3, 0.7],
            output_probs=[0.4, 0.6],
            activation_signature=(1,),
        )
        step = DeepMlpStep(
            x=[0.5, 0.5],
            y=1,
            forward=fwd,
            loss=0.51,
            correct=True,
            grad_norm=0.1,
        )
        assert step.x == [0.5, 0.5]
        assert step.y == 1
        assert step.correct is True


class TestDeepMlpStateDataclass:
    def test_state_serializable(self) -> None:
        state = DeepMlpState(
            input_dim=2,
            hidden_dims=[8],
            output_dim=2,
            lr=0.1,
            hidden_weights=[[[0.1] * 2] * 8],
            hidden_biases=[[0.0] * 8],
            output_weights=[[0.1] * 8, [0.1] * 8],
            output_bias=[0.0, 0.0],
            param_count=34,
        )
        assert state.input_dim == 2
        assert state.hidden_dims == [8]
        assert state.param_count == 34


class TestRegionInfoDataclass:
    def test_region_info(self) -> None:
        info = RegionInfo(count=10, theoretical_max=100, signatures={})
        assert info.count == 10
        assert info.theoretical_max == 100
        assert info.signatures == {}

    def test_region_info_with_signatures(self) -> None:
        sigs = {(1, 0, 1): [(0.0, 0.0), (0.1, 0.1)]}
        info = RegionInfo(count=1, theoretical_max=10, signatures=sigs)
        assert len(info.signatures) == 1


class TestDeepMlpRegionCounting:
    def test_multi_layer_regions(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[8, 8, 8], output_dim=2, seed=42)
        regions = model.count_regions(resolution=20)
        # Multi-layer should have more regions than single layer
        assert regions.count >= 1
        assert regions.theoretical_max > regions.count or regions.count <= regions.theoretical_max

    def test_custom_range(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[4], output_dim=2, seed=42)
        regions = model.count_regions(resolution=10, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5))
        assert regions.count >= 1


class TestDeepMlpTraining:
    def test_xor_learning(self) -> None:
        """Test that the model can learn XOR with enough training."""
        model = DeepMlp(input_dim=2, hidden_dims=[8, 8], output_dim=2, lr=0.5, seed=42)

        # XOR dataset
        samples = [
            ([0.8, 0.8], 0),
            ([0.8, -0.8], 1),
            ([-0.8, 0.8], 1),
            ([-0.8, -0.8], 0),
        ]

        # Train for several epochs
        for _ in range(200):
            for x, y in samples:
                model.step(x, y)

        # Check accuracy
        correct = 0
        for x, y in samples:
            fwd = model.forward(x)
            if fwd.prediction == y:
                correct += 1

        # Should get at least 3/4 correct after training
        assert correct >= 3

    def test_step_returns_valid_grad_norm(self) -> None:
        model = DeepMlp(input_dim=2, hidden_dims=[4, 4], output_dim=2, lr=0.1, seed=42)
        step = model.step([0.5, -0.5], 0)
        assert step.grad_norm >= 0

    def test_different_seeds_different_weights(self) -> None:
        model1 = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2, seed=42)
        model2 = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2, seed=123)
        # Weights should differ
        assert model1.hidden_W[0][0][0] != model2.hidden_W[0][0][0]

    def test_no_hidden_dims_uses_default(self) -> None:
        model = DeepMlp(input_dim=2, output_dim=2)
        assert model.hidden_dims == [16]  # Default

    def test_width_property_empty(self) -> None:
        # Edge case: manually set empty hidden_dims after construction
        model = DeepMlp(input_dim=2, hidden_dims=[8], output_dim=2)
        model.hidden_dims = []  # type: ignore
        assert model.width == 0

