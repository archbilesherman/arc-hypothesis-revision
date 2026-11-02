"""Exact version-space construction and hypothesis supervision targets."""

import torch
from torch import Tensor

from arc_revision.data.schema import (
    GRID_HEIGHT,
    GRID_WIDTH,
    NUM_COLORS,
)
from arc_revision.data.transforms import TRANSFORMS, apply_transform

def consistent_hypotheses(
    input_grid: Tensor,
    output_grid: Tensor,
    ) -> Tensor:
    """
    Identify tranformations consistent with one demonstration.

    Args: 
        input_grid: Input ARC grid [H, W]
        output_grid: Expected output ARC grid [H, W]

    Returns:
        Boolean consistency mask [K]
    """

    for name, grid in (
        ("input_grid", input_grid),
        ("output_grid", output_grid),
    ):
        if not isinstance(grid, Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor")

        if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
            raise ValueError(f"{name} must have shape ({GRID_HEIGHT}, {GRID_WIDTH})")

        if grid.dtype != torch.long:
            raise TypeError(f"{name} must have dtype torch.long")

        if ((grid < 0) | (grid >= NUM_COLORS)).any().item():
            raise ValueError(f"{name} colors must be between 0 and {NUM_COLORS - 1}")

    if input_grid.device != output_grid.device:
        raise ValueError("input_grid and output_grid must be on the same device")

    return torch.tensor(
        [
            torch.equal(apply_transform(input_grid, rule_index), output_grid)
            for rule_index in range(len(TRANSFORMS))
        ],
        dtype=torch.bool,
        device=input_grid.device,
    )

def compute_version_spaces(demonstrations: Tensor) -> Tensor:
    """
    Compute cumulative hypothesis consisterncy after each demo.

    Args:
        demonstrations: Demonstration grids [T, 2, H, W] where index 0 is input and index 1 is output

    Returns:
        Boolean version-space masks [T, K]

    Raises:
        ValueError:- If a version space becomes empty
    """

    if not isinstance(demonstrations, Tensor):
        raise TypeError("demonstrations must be a PyTorch tensor")

    if demonstrations.ndim != 4 or demonstrations.shape[1:] != (2, GRID_HEIGHT, GRID_WIDTH):
        raise ValueError("demonstrations must have shape" f"[T, 2, {GRID_HEIGHT}, {GRID_WIDTH}]")

    if demonstrations.shape[0] == 0:
        raise ValueError("demonstrations must have at least one demonstration")

    if demonstrations.dtype != torch.long:
        raise TypeError("demonstrations must have dtype torch.long")

    if ((demonstrations < 0) | (demonstrations >= NUM_COLORS)).any().item():
        raise ValueError("demonstrations contain invalid colors")

    num_rules = len(TRANSFORMS)

    cumulative_mask = torch.ones(
        num_rules,
        dtype=torch.bool,
        device=demonstrations.device,
        )

    versions_spaces = []

    for t, demonstration in enumerate(demonstrations):
        input_grid = demonstration[0]
        output_grid = demonstration[1]

        current_mask = consistent_hypotheses(input_grid, output_grid)

        # A rule must explain every demonstration seen so far
        cumulative_mask = cumulative_mask & current_mask

        if not cumulative_mask.any().item():
            raise ValueError(f"Version space is empty after demonstration {t + 1}")

        versions_spaces.append(cumulative_mask.clone())

    return torch.stack(versions_spaces, dim=0)

def compute_soft_targets(valid_masks: Tensor) -> Tensor:
    """
    Construct uniform distributions over surviving hypotheses.

    Args:
        valid_masks: Cumulative Boolean masks [T, K]

    Returns:
        Float32 probability distributions [T, K]
    """

    if not isinstance(valid_masks, Tensor):
        raise TypeError("valid_masks must be a PyTorch tensor")

    if (valid_masks.ndim != 2
        or valid_masks.shape[0] == 0
        or valid_masks.shape[1] != len(TRANSFORMS)):
        raise ValueError(f"valid_masks must have shape [T, {len(TRANSFORMS)}] with T >= 1")

    if valid_masks.dtype != torch.bool:
        raise TypeError("valid_masks must have dtype torch.bool")

    counts = valid_masks.sum(dim=1, keepdim=True)

    if (counts == 0).any().item():
        raise ValueError("Cannot construct targets for an empty version space")

    return (valid_masks.to(dtype=torch.float32) / counts.to(dtype=torch.float32))