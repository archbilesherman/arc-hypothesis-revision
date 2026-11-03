"""Tests for parameterized hypothesis programs."""

import pytest
import torch

from arc_revision.data.hypotheses import (
    Hypothesis,
    Operation,
    apply_hypothesis,
    validate_hypothesis_space,
)


@pytest.fixture
def grid():
    return torch.tensor([
        [1, 2, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 4, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 6],
    ], dtype=torch.long)


def test_identity_and_immutability(grid):
    result = apply_hypothesis(grid, Hypothesis())

    assert torch.equal(result, grid)
    assert result.data_ptr() != grid.data_ptr()


def test_ordered_composition(grid):
    program = Hypothesis((
        Operation("rotate", (90,)),
        Operation("replace_color", (1, 7)),
    ))

    expected = torch.rot90(grid, k=-1, dims=(0, 1))
    expected = torch.where(expected == 1, 7, expected)

    assert torch.equal(
        apply_hypothesis(grid, program),
        expected,
    )


def test_replace_and_swap_any_color(grid):
    replace = Hypothesis((
        Operation("replace_color", (2, 9)),
    ))

    swapped = Hypothesis((
        Operation("swap_colors", (1, 2)),
    ))

    assert apply_hypothesis(grid, replace)[0, 1].item() == 9
    assert apply_hypothesis(grid, swapped)[0, 0].item() == 2
    assert apply_hypothesis(grid, swapped)[0, 1].item() == 1
    assert grid[0, 1].item() == 2


def test_full_color_permutation(grid):
    program = Hypothesis((
        Operation("permute_colors", tuple(reversed(range(10)))),
    ))

    result = apply_hypothesis(grid, program)

    assert result[0, 0].item() == 8
    assert result[0, 1].item() == 7
    assert result[0, 2].item() == 9


@pytest.mark.parametrize("kind,params", [
    ("rotate", (45,)),
    ("reflect_horizontal", (1,)),
    ("replace_color", (1, 1)),
    ("swap_colors", (9, 2)),
    ("replace_color", (10, 2)),
    ("permute_colors", (0,) * 10),
    ("unknown", ()),
])
def test_invalid_operations(kind, params):
    with pytest.raises(ValueError):
        Operation(kind, params)


def test_reject_duplicate_programs():
    with pytest.raises(ValueError, match="duplicate"):
        validate_hypothesis_space((
            Hypothesis(),
            Hypothesis(),
        ))


def test_reject_invalid_candidate_space():
    with pytest.raises(ValueError):
        validate_hypothesis_space(())
