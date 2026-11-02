
"""Tests for cumulative version spaces and soft targets."""

import pytest
import torch

from arc_revision.data.version_space import (
    consistent_hypotheses,
    compute_version_spaces,
    compute_soft_targets,
)


def single_cell_grid(row: int, col: int) -> torch.Tensor:
    """Create a 5x5 grid with one colored cell."""
    grid = torch.zeros((5, 5), dtype=torch.long)
    grid[row, col] = 1
    return grid


@pytest.fixture
def ambiguous_demonstrations() -> torch.Tensor:
    """
    Two demonstrations that individually support multiple rules.

    Demo 1: Identity and main-diagonal reflection.
    Demo 2: Identity and anti-diagonal reflection.

    Together, only identity remains consistent.
    """
    first = single_cell_grid(0, 0)
    second = single_cell_grid(0, 4)

    return torch.stack([
        torch.stack([first, first]),
        torch.stack([second, second]),
    ])


def test_individual_consistency():
    """A corner cell is preserved by two D4 transformations."""
    grid = single_cell_grid(0, 0)

    mask = consistent_hypotheses(grid, grid)

    expected = torch.tensor([
        True, False, False, False,
        False, False, True, False,
    ])

    assert mask.dtype == torch.bool
    assert mask.shape == (8,)
    assert torch.equal(mask, expected)


def test_ambiguous_demos_resolve_unique_rule(
    ambiguous_demonstrations,
):
    """Two ambiguous demonstrations jointly identify identity."""
    first_mask = consistent_hypotheses(
        *ambiguous_demonstrations[0]
    )
    second_mask = consistent_hypotheses(
        *ambiguous_demonstrations[1]
    )

    assert first_mask.sum().item() == 2
    assert second_mask.sum().item() == 2

    masks = compute_version_spaces(ambiguous_demonstrations)

    assert masks.shape == (2, 8)
    assert masks[0].sum().item() == 2
    assert masks[1].sum().item() == 1
    assert masks[1, 0].item()
    assert not masks[1, 1:].any().item()


def test_cumulative_version_spaces_cannot_grow(
    ambiguous_demonstrations,
):
    """Eliminated hypotheses cannot reappear."""
    masks = compute_version_spaces(ambiguous_demonstrations)

    for t in range(1, masks.shape[0]):
        assert not (masks[t] & ~masks[t - 1]).any().item()
        assert masks[t].sum() <= masks[t - 1].sum()


def test_soft_targets_normalized(ambiguous_demonstrations):
    """Targets form valid probability distributions."""
    masks = compute_version_spaces(ambiguous_demonstrations)
    targets = compute_soft_targets(masks)

    assert targets.shape == (2, 8)
    assert targets.dtype == torch.float32

    assert torch.allclose(
        targets.sum(dim=-1),
        torch.ones(2),
    )

    assert (targets >= 0).all().item()


def test_soft_targets_uniform(ambiguous_demonstrations):
    """Surviving hypotheses receive equal probability."""
    masks = compute_version_spaces(ambiguous_demonstrations)
    targets = compute_soft_targets(masks)

    expected_first = torch.tensor([
        0.5, 0, 0, 0, 0, 0, 0.5, 0,
    ], dtype=torch.float32)

    expected_second = torch.tensor([
        1, 0, 0, 0, 0, 0, 0, 0,
    ], dtype=torch.float32)

    assert torch.allclose(targets[0], expected_first)
    assert torch.allclose(targets[1], expected_second)


def test_eliminated_hypotheses_have_zero_probability(
    ambiguous_demonstrations,
):
    """Invalid hypotheses always receive zero target weight."""
    masks = compute_version_spaces(ambiguous_demonstrations)
    targets = compute_soft_targets(masks)

    assert torch.all(targets[~masks] == 0).item()


def test_empty_version_space():
    """Impossible demonstrations produce a clear error."""
    input_grid = torch.zeros((5, 5), dtype=torch.long)
    output_grid = torch.ones((5, 5), dtype=torch.long)

    demos = torch.stack([
        torch.stack([input_grid, output_grid]),
    ])

    with pytest.raises(ValueError, match="Version space is empty"):
        compute_version_spaces(demos)


def test_empty_intersection():
    """Individually valid but contradictory demos are rejected."""
    grid = single_cell_grid(0, 0)

    # First demo requires identity or main-diagonal reflection.
    first = torch.stack([grid, grid])

    # Second demo requires a 180-degree rotation or
    # anti-diagonal reflection.
    rotated = torch.rot90(grid, k=2, dims=(0, 1))
    second = torch.stack([grid, rotated])

    demos = torch.stack([first, second])

    with pytest.raises(ValueError, match="demonstration 2"):
        compute_version_spaces(demos)


def test_empty_soft_target_mask():
    """Empty hypothesis sets cannot be normalized."""
    masks = torch.zeros((2, 8), dtype=torch.bool)

    with pytest.raises(ValueError, match="empty version space"):
        compute_soft_targets(masks)


def test_invalid_demonstration_shape():
    demos = torch.zeros((2, 2, 4, 5), dtype=torch.long)

    with pytest.raises(ValueError):
        compute_version_spaces(demos)


def test_invalid_demonstration_dtype():
    demos = torch.zeros((2, 2, 5, 5), dtype=torch.float32)

    with pytest.raises(TypeError):
        compute_version_spaces(demos)


@pytest.mark.parametrize("color", [-1, 10])
def test_invalid_demonstration_colors(color):
    demos = torch.zeros((2, 2, 5, 5), dtype=torch.long)
    demos[0, 0, 0, 0] = color

    with pytest.raises(ValueError):
        compute_version_spaces(demos)


def test_zero_demonstrations():
    demos = torch.empty((0, 2, 5, 5), dtype=torch.long)

    with pytest.raises(ValueError):
        compute_version_spaces(demos)


def test_invalid_soft_target_shape():
    masks = torch.ones((2, 7), dtype=torch.bool)

    with pytest.raises(ValueError):
        compute_soft_targets(masks)


def test_invalid_soft_target_dtype():
    masks = torch.ones((2, 8), dtype=torch.float32)

    with pytest.raises(TypeError):
        compute_soft_targets(masks)
