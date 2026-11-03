"""Unit tests for the D4 transformation library."""

import pytest
import torch

from arc_revision.data.transforms import (
    TRANSFORM_NAMES,
    TRANSFORMS,
    apply_transform,
)


@pytest.fixture
def asymmetric_grid() -> torch.Tensor:
    """A 5x5 grid with no nontrivial D4 symmetry."""
    return torch.tensor([
        [1, 2, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 4, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 6],
    ], dtype=torch.long)


def test_registry():
    """Registry contains eight uniquely named transformations."""
    assert len(TRANSFORMS) == 8
    assert len(TRANSFORM_NAMES) == 8
    assert len(set(TRANSFORM_NAMES)) == 8


@pytest.mark.parametrize("rule_index", range(8))
def test_preserves_shape_dtype_and_colors(asymmetric_grid, rule_index):
    """Transformations preserve grid dimensions, dtype, and colors."""
    result = apply_transform(asymmetric_grid, rule_index)

    assert result.shape == asymmetric_grid.shape
    assert result.dtype == asymmetric_grid.dtype
    assert torch.equal(
        torch.sort(result.flatten()).values,
        torch.sort(asymmetric_grid.flatten()).values,
    )


@pytest.mark.parametrize("rule_index", range(8))
def test_does_not_mutate_input(asymmetric_grid, rule_index):
    """Transformations must not modify their input tensors."""
    original = asymmetric_grid.clone()

    result = apply_transform(asymmetric_grid, rule_index)

    assert torch.equal(asymmetric_grid, original)
    assert result.data_ptr() != asymmetric_grid.data_ptr()


def test_all_transformations_distinct(asymmetric_grid):
    """All eight D4 transformations differ on an asymmetric grid."""
    results = [
        apply_transform(asymmetric_grid, i)
        for i in range(8)
    ]

    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.equal(results[i], results[j]), (
                f"Transforms {TRANSFORM_NAMES[i]} and "
                f"{TRANSFORM_NAMES[j]} produced identical outputs"
            )


@pytest.mark.parametrize(
    "first,second",
    [
        (1, 3),  # 90 + 270
        (3, 1),  # 270 + 90
        (2, 2),  # 180 + 180
        (4, 4),  # Horizontal reflection twice
        (5, 5),  # Vertical reflection twice
        (6, 6),  # Main diagonal reflection twice
        (7, 7),  # Anti-diagonal reflection twice
    ],
)
def test_inverse_properties(asymmetric_grid, first, second):
    """Inverse operations recover the original grid."""
    result = apply_transform(
        apply_transform(asymmetric_grid, first),
        second,
    )

    assert torch.equal(result, asymmetric_grid)


def test_rotation_composition(asymmetric_grid):
    """Two 90-degree rotations equal a 180-degree rotation."""
    twice = apply_transform(
        apply_transform(asymmetric_grid, 1), 1
    )
    once = apply_transform(asymmetric_grid, 2)

    assert torch.equal(twice, once)


def test_reflection_composition(asymmetric_grid):
    """Horizontal and vertical reflections compose to 180 degrees."""
    result = apply_transform(
        apply_transform(asymmetric_grid, 4), 5
    )
    expected = apply_transform(asymmetric_grid, 2)

    assert torch.equal(result, expected)


@pytest.mark.parametrize("rule_index", [-1, 8, 10])
def test_invalid_rule_index(asymmetric_grid, rule_index):
    with pytest.raises(ValueError):
        apply_transform(asymmetric_grid, rule_index)


def test_invalid_rule_index_type(asymmetric_grid):
    with pytest.raises(TypeError):
        apply_transform(asymmetric_grid, 1.5)


def test_invalid_grid_shape():
    grid = torch.zeros((4, 5), dtype=torch.long)

    with pytest.raises(ValueError):
        apply_transform(grid, 0)


def test_invalid_grid_dtype():
    grid = torch.zeros((5, 5), dtype=torch.float32)

    with pytest.raises(TypeError):
        apply_transform(grid, 0)


@pytest.mark.parametrize("color", [-1, 10])
def test_invalid_colors(asymmetric_grid, color):
    asymmetric_grid[0, 0] = color

    with pytest.raises(ValueError):
        apply_transform(asymmetric_grid, 0)