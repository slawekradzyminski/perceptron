"""Datasets for perceptron demos."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


Sample = dict


def _pm1(val: int) -> int:
    return 1 if val > 0 else -1


def _flatten(grid: Sequence[Sequence[int]]) -> List[int]:
    return [cell for row in grid for cell in row]


def make_or_dataset_pm1() -> List[Sample]:
    # Inputs are in {-1, +1}. OR is positive if any input is +1.
    data = [
        ([-1, -1], -1),
        ([-1, 1], 1),
        ([1, -1], 1),
        ([1, 1], 1),
    ]
    return [{"x": x, "y": y} for x, y in data]


def make_and_dataset_pm1() -> List[Sample]:
    # AND is positive only if both inputs are +1.
    data = [
        ([-1, -1], -1),
        ([-1, 1], -1),
        ([1, -1], -1),
        ([1, 1], 1),
    ]
    return [{"x": x, "y": y} for x, y in data]


def make_xor_dataset_pm1() -> List[Sample]:
    # XOR is positive if exactly one input is +1.
    data = [
        ([-1, -1], -1),
        ([-1, 1], 1),
        ([1, -1], 1),
        ([1, 1], -1),
    ]
    return [{"x": x, "y": y} for x, y in data]


def generate_translations(
    shape_mask: Sequence[Sequence[int]],
    board_h: int,
    board_w: int,
) -> List[Tuple[List[List[int]], Tuple[int, int]]]:
    """Return all valid translations of a shape on a board.

    Cells belonging to the shape are +1, empty cells are -1.
    Returns list of (grid, (top, left)).
    """
    shape_h = len(shape_mask)
    shape_w = len(shape_mask[0]) if shape_h > 0 else 0
    if shape_h == 0 or shape_w == 0:
        raise ValueError("shape_mask must be non-empty")
    if board_h < shape_h or board_w < shape_w:
        raise ValueError("board must be at least as large as shape")

    positions: List[Tuple[List[List[int]], Tuple[int, int]]] = []
    for top in range(board_h - shape_h + 1):
        for left in range(board_w - shape_w + 1):
            grid = [[-1 for _ in range(board_w)] for _ in range(board_h)]
            for r in range(shape_h):
                for c in range(shape_w):
                    if _pm1(shape_mask[r][c]) == 1:
                        grid[top + r][left + c] = 1
            positions.append((grid, (top, left)))
    return positions


def make_shape_dataset(
    good_mask: Sequence[Sequence[int]],
    bad_mask: Sequence[Sequence[int]],
    board_size: Tuple[int, int],
    translations: bool = True,
) -> List[Sample]:
    """Create a dataset of good vs bad shapes on a grid.

    Each sample includes:
    - x: flattened grid in {-1, +1}
    - y: label (+1 for good, -1 for bad)
    - grid: 2D grid for visualization
    - pos: (top, left) if translations are used
    """
    board_h, board_w = board_size
    samples: List[Sample] = []

    def _add(mask: Sequence[Sequence[int]], label: int) -> None:
        if translations:
            for grid, pos in generate_translations(mask, board_h, board_w):
                samples.append({"x": _flatten(grid), "y": label, "grid": grid, "pos": pos})
        else:
            if len(mask) != board_h or len(mask[0]) != board_w:
                raise ValueError("mask must match board size when translations=False")
            grid = [[_pm1(cell) for cell in row] for row in mask]
            samples.append({"x": _flatten(grid), "y": label, "grid": grid, "pos": (0, 0)})

    _add(good_mask, 1)
    _add(bad_mask, -1)
    return samples
