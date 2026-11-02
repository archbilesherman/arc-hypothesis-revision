
"""Deterministic D4 transformations for ARC grids."""

from collections.abc import Callable

import torch
from torch import Tensor


def identity(grid: Tensor) -> Tensor:
    """Return the grid unchanged."""
    return grid.clone()


def rotate_90(grid: Tensor) -> Tensor:
    """Rotate 90 degrees clockwise."""
    return torch.rot90(grid, k=-1, dims=(0, 1)).clone()


def rotate_180(grid: Tensor) -> Tensor:
    """Rotate 180 degrees."""
    return torch.rot90(grid, k=2, dims=(0, 1)).clone()


def rotate_270(grid: Tensor) -> Tensor:
    """Rotate 270 degrees clockwise."""
    return torch.rot90(grid, k=1, dims=(0, 1)).clone()


def reflect_horizontal(grid: Tensor) -> Tensor:
    """Reflect across the horizontal axis (flip top/bottom)."""
    return torch.flip(grid, dims=(0,))


def reflect_vertical(grid: Tensor) -> Tensor:
    """Reflect across the vertical axis (flip left/right)."""
    return torch.flip(grid, dims=(1,))


def reflect_main_diagonal(grid: Tensor) -> Tensor:
    """Reflect across the top-left to bottom-right diagonal."""
    return grid.transpose(0, 1).clone()


def reflect_anti_diagonal(grid: Tensor) -> Tensor:
    """Reflect across the top-right to bottom-left diagonal."""
    return torch.flip(grid.transpose(0, 1), dims=(0, 1))


# Stable ordering: indices serve as hypothesis classification labels.
TRANSFORM_NAMES: tuple[str, ...] = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "reflect_horizontal",
    "reflect_vertical",
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
)

TRANSFORMS: tuple[Callable[[Tensor], Tensor], ...] = (
    identity,
    rotate_90,
    rotate_180,
    rotate_270,
    reflect_horizontal,
    reflect_vertical,
    reflect_main_diagonal,
    reflect_anti_diagonal,
)


def apply_transform(grid: Tensor, rule_index: int) -> Tensor:
    """
    Apply a transformation by its canonical rule index.

    Args:
        grid: A square 2D integer tensor with ARC colors 0-9.
        rule_index: Transformation index in [0, 7].

    Returns:
        Transformed grid with the same shape, dtype, and colors.

    Raises:
        TypeError: Invalid grid type, dtype, or rule index type.
        ValueError: Invalid grid dimensions, colors, or rule index.
    """
    if not isinstance(grid, Tensor):
        raise TypeError("grid must be a PyTorch tensor")

    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("grid must be a square 2D tensor")

    if grid.dtype != torch.long:
        raise TypeError("grid must have dtype torch.long")

    if ((grid < 0) | (grid > 9)).any().item():
        raise ValueError("grid colors must be between 0 and 9")

    if type(rule_index) is not int:
        raise TypeError("rule_index must be an integer")

    if not 0 <= rule_index < len(TRANSFORMS):
        raise ValueError(
            f"rule_index must be between 0 and {len(TRANSFORMS) - 1}"
        )

    return TRANSFORMS[rule_index](grid)
