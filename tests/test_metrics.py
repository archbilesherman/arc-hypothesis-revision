import math

import pytest
import torch

from arc_revision.data.hypotheses import (
    Hypothesis,
    Operation,
    apply_hypothesis,
)
from arc_revision.evaluation.metrics import (
    final_candidate_indices,
    grid_exact_matches,
    program_correct,
    valid_space_mass,
)


def test_final_candidate_selection_uses_last_real_step():
    logits = torch.tensor([
        [
            [8.0, 0.0, 100.0],
            [2.0, 5.0, 100.0],
            [999.0, 0.0, 0.0],
        ],
        [
            [0.0, 8.0, 100.0],
            [1.0, 5.0, 100.0],
            [4.0, 1.0, 100.0],
        ],
    ])

    demo_mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    candidate_mask = torch.tensor([
        [True, True, False],
        [True, True, False],
    ])

    selected = final_candidate_indices(
        logits, demo_mask, candidate_mask
    )

    assert selected.tolist() == [1, 0]


def test_program_correctness_known_values():
    predicted = torch.tensor([1, 0, 2, 3])
    targets = torch.tensor([1, 2, 2, 0])

    correct = program_correct(predicted, targets)

    assert correct.tolist() == [
        True, False, True, False
    ]
    assert correct.float().mean().item() == 0.5


def test_query_exact_match_requires_entire_grid():
    predicted = torch.tensor([
        [[1, 2], [3, 4]],
        [[0, 0], [0, 1]],
    ])
    targets = torch.tensor([
        [[1, 2], [3, 4]],
        [[0, 0], [0, 0]],
    ])

    result = grid_exact_matches(predicted, targets)

    assert result.tolist() == [True, False]


def test_wrong_program_can_produce_correct_grid():
    identity = Hypothesis()
    rotation = Hypothesis((
        Operation("rotate", (180,)),
    ))

    grid = torch.zeros(5, 5, dtype=torch.long)

    predicted = apply_hypothesis(grid, rotation)
    expected = apply_hypothesis(grid, identity)

    assert not program_correct(
        torch.tensor([1]), torch.tensor([0])
    ).item()

    assert grid_exact_matches(
        predicted.unsqueeze(0),
        expected.unsqueeze(0),
    ).item()


def test_known_version_space_probability_mass():
    logits = torch.tensor([
        [
            [math.log(3), 0.0, 1000.0],
            [math.log(3), 0.0, 1000.0],
            [float("nan"), float("nan"), float("nan")],
        ],
    ], dtype=torch.float64)

    valid_masks = torch.tensor([
        [
            [True, True, False],
            [False, True, False],
            [False, False, False],
        ],
    ])

    demo_mask = torch.tensor([
        [True, True, False]
    ])
    candidate_mask = torch.tensor([
        [True, True, False]
    ])

    mass = valid_space_mass(
        logits,
        valid_masks,
        demo_mask,
        candidate_mask,
    )

    torch.testing.assert_close(
        mass,
        torch.tensor([[1.0, 0.25, 0.0]], dtype=torch.float64),
    )


def test_padded_candidates_cannot_win_selection():
    logits = torch.tensor([
        [[1.0, 2.0, 1e20]],
    ])

    result = final_candidate_indices(
        logits,
        torch.tensor([[True]]),
        torch.tensor([[True, True, False]]),
    )

    assert result.item() == 1


def test_padded_candidate_values_do_not_change_consistency():
    logits = torch.tensor([
        [[1.0, 0.0, 1e20]],
    ])
    valid = torch.tensor([
        [[True, False, False]],
    ])
    demos = torch.tensor([[True]])
    candidates = torch.tensor([[True, True, False]])

    modified = logits.clone()
    modified[0, 0, 2] = float("nan")

    original_mass = valid_space_mass(
        logits, valid, demos, candidates
    )
    changed_mass = valid_space_mass(
        modified, valid, demos, candidates
    )

    torch.testing.assert_close(original_mass, changed_mass)


def test_nonprefix_demonstration_mask_rejected():
    logits = torch.zeros(1, 3, 2)
    demo_mask = torch.tensor([
        [True, False, True]
    ])
    candidate_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="prefix"):
        final_candidate_indices(
            logits, demo_mask, candidate_mask
        )


def test_empty_candidate_set_rejected():
    logits = torch.zeros(1, 2, 2)
    demo_mask = torch.ones(1, 2, dtype=torch.bool)
    candidate_mask = torch.zeros(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="candidates"):
        final_candidate_indices(
            logits, demo_mask, candidate_mask
        )


def test_invalid_grid_shapes_rejected():
    with pytest.raises(ValueError, match="shape"):
        grid_exact_matches(
            torch.zeros(2, 5, 5),
            torch.zeros(2, 4, 5),
        )