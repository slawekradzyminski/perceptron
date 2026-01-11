"""Diagnostics service for error surface and MLP internals visualization."""

from __future__ import annotations

from typing import Any

from backend.nn.grid_mlp import GridMlp, reshape_template
from backend.viz.viz_error_surface import mse_surface


class DiagnosticsService:
    """Service for stateless diagnostic computations.

    Unlike stateful services (PerceptronService, MlpService), this service
    performs one-shot computations without maintaining state between requests.
    """

    def error_surface(
        self,
        samples: list[dict[str, Any]],
        w_min: float,
        w_max: float,
        steps: int,
        b: float,
    ) -> list[list[float]]:
        """Compute MSE error surface over a weight grid.

        Args:
            samples: Training samples with 'x' and 'y' keys
            w_min: Minimum weight value
            w_max: Maximum weight value
            steps: Grid resolution
            b: Fixed bias value

        Returns:
            2D grid of MSE values
        """
        return mse_surface(samples, (w_min, w_max), steps=steps, b=b)

    def mlp_internals(
        self,
        samples: list[dict[str, Any]],
        rows: int,
        cols: int,
        hidden_dim: int,
        lr: float,
        seed: int,
        sample_index: int,
    ) -> dict[str, Any]:
        """Compute MLP internals for a single forward-backward pass.

        Args:
            samples: Training samples with 'x' and 'y' keys
            rows: Grid rows (for template visualization)
            cols: Grid cols (for template visualization)
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            seed: Random seed
            sample_index: Which sample to inspect

        Returns:
            Dictionary with layer activations, gradients, and templates
        """
        sample = samples[sample_index % len(samples)]
        model = GridMlp(rows=rows, cols=cols, hidden_dim=hidden_dim, lr=lr, seed=seed)
        internals = model.model.inspect_step(sample["x"], sample["y"])

        # Build templates for visualization
        templates_before = [reshape_template(row, rows, cols) for row in internals.hidden_W_before]
        templates_after = [reshape_template(row, rows, cols) for row in internals.hidden_W_after]
        gradient_templates = [reshape_template(row, rows, cols) for row in internals.grad_hidden_W]

        return {
            "dataset": "custom",
            "grid_rows": rows,
            "grid_cols": cols,
            "hidden_dim": hidden_dim,
            "sample_index": sample_index % len(samples),
            "sample_count": len(samples),
            "x": internals.x,
            "y": internals.y,
            "y01": internals.y01,
            "loss": internals.loss,
            "p_hat": internals.output_a,
            "hidden": {
                "weights_before": internals.hidden_W_before,
                "bias_before": internals.hidden_b_before,
                "weights_after": internals.hidden_W_after,
                "bias_after": internals.hidden_b_after,
                "z": internals.hidden_z,
                "a": internals.hidden_a,
                "templates_before": templates_before,
                "templates_after": templates_after,
            },
            "output": {
                "weights_before": internals.out_W_before,
                "bias_before": internals.out_b_before,
                "weights_after": internals.out_W_after,
                "bias_after": internals.out_b_after,
                "z": internals.output_z,
                "a": internals.output_a,
            },
            "gradients": {
                "hidden_W": internals.grad_hidden_W,
                "hidden_b": internals.grad_hidden_b,
                "out_W": internals.grad_out_W,
                "out_b": internals.grad_out_b,
                "templates": gradient_templates,
            },
        }
