"""Tests for version spaces over executable hypothesis programs."""

import pytest
import torch

from arc_revision.data.hypotheses import Hypothesis, Operation
from arc_revision.data.version_space import (
    consistent_hypotheses,
    compute_version_spaces,
    compute_soft_targets,
)


@pytest.fixture
def candidates():
    return (
        Hypothesis(),
        Hypothesis((Operation("reflect_main_diagonal"),)),
        Hypothesis((Operation("reflect_anti_diagonal"),)),
        Hypothesis((Operation("rotate", (180,)),)),
        Hypothesis((Operation("replace_color", (1, 2)),)),
    )


def single_cell_grid(row: int, col: int) -> torch.Tensor:
    grid = torch.zeros((5, 5), dtype=torch.long)
    grid[row, col] = 1
    return grid


def test_individual_and_cumulative_consistency(candidates):
    first = single_cell_grid(0, 0)
    second = single_cell_grid(0, 4)

    demos = torch.stack([
        torch.stack([first, first]),
        torch.stack([second, second]),
    ])

    first_mask = consistent_hypotheses(
        first, first, candidates
    )

    second_mask = consistent_hypotheses(
        second, second, candidates
    )

    assert first_mask.tolist() == [
        True, True, False, False, False
    ]
    assert second_mask.tolist() == [
        True, False, True, False, False
    ]

    masks = compute_version_spaces(demos, candidates)

    assert masks.shape == (2, len(candidates))
    assert masks.sum(dim=-1).tolist() == [2, 1]
    assert masks[-1, 0].item()

    assert not (
        masks[1:] & ~masks[:-1]
    ).any().item()

    targets = compute_soft_targets(masks)

    assert torch.allclose(
        targets[0],
        torch.tensor([0.5, 0.5, 0, 0, 0]),
    )

    assert torch.allclose(
        targets[1],
        torch.tensor([1., 0, 0, 0, 0]),
    )


def test_five_informative_demonstrations():
    """Five demonstrations progressively eliminate five candidates."""

    candidates = (Hypothesis(),) + tuple(
        Hypothesis((
            Operation("replace_color", (color, 9)),
        ))
        for color in range(1, 6)
    )

    demos = []

    for color in range(1, 6):
        grid = single_cell_grid(0, 0)
        grid[0, 0] = color

        demos.append(torch.stack([grid, grid]))

    masks = compute_version_spaces(
        torch.stack(demos),
        candidates,
    )

    assert masks.shape == (5, 6)
    assert masks.sum(dim=-1).tolist() == [
        5, 4, 3, 2, 1
    ]
    assert masks[-1, 0].item()


def test_empty_version_space(candidates):
    demos = torch.stack([
        torch.stack([
            torch.zeros((5, 5), dtype=torch.long),
            torch.ones((5, 5), dtype=torch.long),
        ])
    ])

    with pytest.raises(
        ValueError,
        match="Version space is empty",
    ):
        compute_version_spaces(demos, candidates)


def test_empty_intersection(candidates):
    grid = single_cell_grid(0, 0)
    rotated = torch.rot90(grid, k=2, dims=(0, 1))

    demos = torch.stack([
        torch.stack([grid, grid]),
        torch.stack([grid, rotated]),
    ])

    with pytest.raises(ValueError, match="demonstration 2"):
        compute_version_spaces(demos, candidates)


def test_soft_target_rejects_empty_mask():
    with pytest.raises(
        ValueError,
        match="empty version space",
    ):
        compute_soft_targets(
            torch.zeros((2, 3), dtype=torch.bool)
        )


def test_soft_target_rejects_zero_candidates():
    with pytest.raises(ValueError, match="shape"):
        compute_soft_targets(
            torch.zeros((2, 0), dtype=torch.bool)
        )


def test_soft_target_arbitrary_k():
    masks = torch.tensor([
        [True, False, True],
        [True, False, False],
    ])

    assert compute_soft_targets(masks).shape == (2, 3)


def test_invalid_demo_shape(candidates):
    demos = torch.zeros((2, 2, 4, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="shape"):
        compute_version_spaces(demos, candidates)


def test_invalid_dtype(candidates):
    demos = torch.zeros((2, 2, 5, 5))

    with pytest.raises(TypeError):
        compute_version_spaces(demos, candidates)
