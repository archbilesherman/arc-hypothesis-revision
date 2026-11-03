"""Exact version-space construction and hypothesis supervision targets."""

import torch
from torch import Tensor

from arc_revision.data.hypotheses import (
    Hypothesis,
    apply_hypothesis,
    validate_hypothesis_space,
)

from arc_revision.data.schema import (
    GRID_HEIGHT,
    GRID_WIDTH,
    NUM_COLORS,
)


def consistent_hypotheses(
    input_grid: Tensor,
    output_grid: Tensor,
    hypotheses: tuple[Hypothesis, ...],
) -> Tensor:
    """Return a Boolean mask [K] of programs consistent with one pair."""

    validate_hypothesis_space(hypotheses)

    for name, grid in (
        ("input_grid", input_grid),
        ("output_grid", output_grid),
    ):
        if not isinstance(grid, Tensor):
            raise TypeError(
                f"{name} must be a PyTorch tensor"
            )

        if tuple(grid.shape) != (GRID_HEIGHT, GRID_WIDTH):
            raise ValueError(
                f"{name} must have shape "
                f"({GRID_HEIGHT}, {GRID_WIDTH})"
            )

        if grid.dtype != torch.long:
            raise TypeError(
                f"{name} must have dtype torch.long"
            )

        if ((grid < 0) | (grid >= NUM_COLORS)).any().item():
            raise ValueError(
                f"{name} colors must be between 0 and {NUM_COLORS - 1}"
            )

    if input_grid.device != output_grid.device:
        raise ValueError(
            "input_grid and output_grid must be on the same device"
        )

    return torch.tensor(
        [
            torch.equal(
                apply_hypothesis(input_grid, hypothesis),
                output_grid,
            )
            for hypothesis in hypotheses
        ],
        dtype=torch.bool,
        device=input_grid.device,
    )


def compute_version_spaces(
    demonstrations: Tensor,
    hypotheses: tuple[Hypothesis, ...],
) -> Tensor:
    """Accumulate consistency masks across T demonstrations: [T, K]."""

    validate_hypothesis_space(hypotheses)

    if not isinstance(demonstrations, Tensor):
        raise TypeError(
            "demonstrations must be a PyTorch tensor"
        )

    if (
        demonstrations.ndim != 4
        or tuple(demonstrations.shape[1:])
        != (2, GRID_HEIGHT, GRID_WIDTH)
    ):
        raise ValueError(
            f"demonstrations must have shape "
            f"[T, 2, {GRID_HEIGHT}, {GRID_WIDTH}]"
        )

    if demonstrations.shape[0] == 0:
        raise ValueError(
            "demonstrations must have at least one demonstration"
        )

    if demonstrations.dtype != torch.long:
        raise TypeError(
            "demonstrations must have dtype torch.long"
        )

    if (
        (demonstrations < 0)
        | (demonstrations >= NUM_COLORS)
    ).any().item():
        raise ValueError(
            "demonstrations contain invalid colors"
        )

    cumulative_mask = torch.ones(
        len(hypotheses),
        dtype=torch.bool,
        device=demonstrations.device,
    )

    version_spaces = []

    for t, (input_grid, output_grid) in enumerate(demonstrations):
        current_mask = consistent_hypotheses(
            input_grid,
            output_grid,
            hypotheses,
        )

        cumulative_mask = cumulative_mask & current_mask

        if not cumulative_mask.any().item():
            raise ValueError(
                f"Version space is empty after demonstration {t + 1}"
            )

        version_spaces.append(cumulative_mask.clone())

    return torch.stack(version_spaces, dim=0)


def compute_soft_targets(valid_masks: Tensor) -> Tensor:
    """Normalize surviving hypotheses uniformly: [T, K] float32."""

    if not isinstance(valid_masks, Tensor):
        raise TypeError(
            "valid_masks must be a PyTorch tensor"
        )

    if (
        valid_masks.ndim != 2
        or valid_masks.shape[0] == 0
        or valid_masks.shape[1] == 0
    ):
        raise ValueError(
            "valid_masks must have shape [T, K] with T,K >= 1"
        )

    if valid_masks.dtype != torch.bool:
        raise TypeError(
            "valid_masks must have dtype torch.bool"
        )

    counts = valid_masks.sum(dim=-1, keepdim=True)

    if (counts == 0).any().item():
        raise ValueError(
            "Cannot construct targets for an empty version space"
        )

    return (
        valid_masks.to(dtype=torch.float32)
        / counts.to(dtype=torch.float32)
    )
