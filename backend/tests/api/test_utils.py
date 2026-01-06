"""Tests for backend.api.utils helpers."""

import pytest

from backend.api.utils import (
    load_samples_from_body,
    normalize_samples,
    validate_grid_shape,
)


def test_validate_grid_shape_parses_ints() -> None:
    rows, cols = validate_grid_shape("2", 3)
    assert rows == 2
    assert cols == 3


def test_validate_grid_shape_invalid_type() -> None:
    with pytest.raises(ValueError, match="grid_rows and grid_cols must be integers"):
        validate_grid_shape(None, "bad")


def test_validate_grid_shape_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_grid_shape(0, 2)
    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_grid_shape(1, 6)


def test_normalize_samples_from_grid() -> None:
    samples = [{"grid": [[1, -1]], "y": 1}]
    normalized = normalize_samples(samples, rows=1, cols=2)
    assert normalized == [{"x": [1.0, -1.0], "y": 1}]


def test_normalize_samples_from_x() -> None:
    samples = [{"x": [1, -1], "y": -1}]
    normalized = normalize_samples(samples, rows=1, cols=2)
    assert normalized == [{"x": [1.0, -1.0], "y": -1}]


def test_normalize_samples_rejects_invalid_y() -> None:
    with pytest.raises(ValueError, match=r"y must be -1 or \+1"):
        normalize_samples([{"x": [1, -1], "y": 0}], rows=1, cols=2)


def test_normalize_samples_rejects_bad_grid_shape() -> None:
    with pytest.raises(ValueError, match="grid rows must match"):
        normalize_samples([{"grid": [[1, -1], [1, 1]], "y": 1}], rows=1, cols=2)


def test_normalize_samples_rejects_bad_grid_values() -> None:
    with pytest.raises(ValueError, match=r"grid values must be -1 or \+1"):
        normalize_samples([{"grid": [[0, 1]], "y": 1}], rows=1, cols=2)


def test_normalize_samples_rejects_bad_x_length() -> None:
    with pytest.raises(ValueError, match="x length must match"):
        normalize_samples([{"x": [1, -1, 1], "y": 1}], rows=1, cols=2)


def test_normalize_samples_rejects_bad_x_values() -> None:
    with pytest.raises(ValueError, match=r"x values must be -1 or \+1"):
        normalize_samples([{"x": [1, 0], "y": 1}], rows=1, cols=2)


def test_normalize_samples_rejects_empty() -> None:
    with pytest.raises(ValueError, match="samples must be non-empty"):
        normalize_samples([], rows=1, cols=2)


def test_load_samples_from_body_default_or() -> None:
    dataset, samples, shape = load_samples_from_body({})
    assert dataset == "or"
    assert shape == (1, 2)
    assert len(samples) == 4


def test_load_samples_from_body_xor() -> None:
    dataset, samples, shape = load_samples_from_body({"dataset": "xor"})
    assert dataset == "xor"
    assert shape == (1, 2)
    assert len(samples) == 4


def test_load_samples_from_body_custom() -> None:
    body = {
        "dataset": "custom",
        "grid_rows": 1,
        "grid_cols": 2,
        "samples": [{"grid": [[1, -1]], "y": 1}],
    }
    dataset, samples, shape = load_samples_from_body(body)
    assert dataset == "custom"
    assert shape == (1, 2)
    assert samples == [{"x": [1.0, -1.0], "y": 1}]


def test_load_samples_from_body_invalid_dataset() -> None:
    with pytest.raises(ValueError, match="dataset must be"):
        load_samples_from_body({"dataset": "bad"})
