from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from backend.core.datasets import make_or_dataset_pm1, make_xor_dataset_pm1
from backend.nn.grid_mlp import reshape_template
from backend.nn.mlp import MlpInternals


def validate_grid_shape(rows: Any, cols: Any) -> Tuple[int, int]:
    try:
        r = int(rows)
        c = int(cols)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_rows and grid_cols must be integers") from exc
    if r < 1 or c < 1 or r > 5 or c > 5:
        raise ValueError("grid_rows and grid_cols must be between 1 and 5")
    return r, c


def normalize_samples(
    samples: Iterable[Dict[str, Any]],
    rows: int,
    cols: int,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for sample in samples:
        y = sample.get("y")
        if y not in (-1, 1):
            raise ValueError("each sample y must be -1 or +1")
        grid = sample.get("grid")
        x = sample.get("x")
        if grid is not None:
            if not isinstance(grid, list) or len(grid) != rows:
                raise ValueError("grid rows must match grid_rows")
            flat: List[float] = []
            for row in grid:
                if not isinstance(row, list) or len(row) != cols:
                    raise ValueError("grid cols must match grid_cols")
                for cell in row:
                    if cell not in (-1, 1):
                        raise ValueError("grid values must be -1 or +1")
                    flat.append(float(cell))
            x = flat
        if x is None:
            raise ValueError("each sample must include grid or x")
        if not isinstance(x, list) or len(x) != rows * cols:
            raise ValueError("x length must match grid_rows * grid_cols")
        for cell in x:
            if cell not in (-1, 1):
                raise ValueError("x values must be -1 or +1")
        normalized.append({"x": [float(val) for val in x], "y": int(y)})
    if not normalized:
        raise ValueError("samples must be non-empty")
    return normalized


def load_samples_from_body(body: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Tuple[int, int]]:
    dataset = body.get("dataset", "or")
    if dataset == "or":
        return "or", make_or_dataset_pm1(), (1, 2)
    if dataset == "xor":
        return "xor", make_xor_dataset_pm1(), (1, 2)
    if dataset == "custom":
        rows, cols = validate_grid_shape(body.get("grid_rows"), body.get("grid_cols"))
        samples = normalize_samples(body.get("samples", []), rows, cols)
        return "custom", samples, (rows, cols)
    raise ValueError("dataset must be 'or', 'xor', or 'custom'")


def build_mlp_payload(
    internals: MlpInternals,
    rows: int,
    cols: int,
    hidden_dim: int,
    sample_index: int,
    sample_count: int,
    dataset: str,
) -> Dict[str, Any]:
    templates_before = [reshape_template(row, rows, cols) for row in internals.hidden_W_before]
    templates_after = [reshape_template(row, rows, cols) for row in internals.hidden_W_after]
    gradient_templates = [reshape_template(row, rows, cols) for row in internals.grad_hidden_W]
    return {
        "dataset": dataset,
        "grid_rows": rows,
        "grid_cols": cols,
        "hidden_dim": hidden_dim,
        "sample_index": sample_index,
        "sample_count": sample_count,
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
