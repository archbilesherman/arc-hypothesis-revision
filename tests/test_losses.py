import math

import pytest
import torch
from torch.nn import functional as F

from arc_revision.data.dataset import (
    ARCEpisodeDataset,
    collate_episodes,
)
from arc_revision.data.generator import generate_episode
from arc_revision.models.recurrent import (
    SequentialGRUHypothesisClassifier,
)
from arc_revision.training.losses import (
    final_only_loss,
    revision_aware_loss,
    revision_loss,
)


@pytest.fixture
def example():
    """Two episodes with different T and K and uniform initial logits."""
    logits = torch.zeros(2, 3, 4, dtype=torch.float64)

    demo_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1]],
        dtype=torch.bool,
    )
    candidate_mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    rule_index = torch.tensor([1, 2], dtype=torch.long)

    soft_targets = torch.tensor(
        [
            [
                [0.5, 0.5, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0.25, 0.25, 0.25, 0.25],
                [0, 0.5, 0.5, 0],
                [0, 0, 1, 0],
            ],
        ],
        dtype=torch.float64,
    )

    return (
        logits,
        rule_index,
        demo_mask,
        candidate_mask,
        soft_targets,
    )


def test_uniform_final_known_value(example):
    logits, rules, demos, candidates, _ = example

    expected = (math.log(2) + math.log(4)) / 2

    torch.testing.assert_close(
        final_only_loss(logits, rules, demos, candidates),
        torch.tensor(expected, dtype=logits.dtype),
    )


def test_uniform_revision_known_value_and_episode_averaging(example):
    logits, _, demos, candidates, targets = example

    expected = (math.log(2) + math.log(4)) / 2
    actual = revision_loss(logits, targets, demos, candidates)

    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=logits.dtype),
    )

    # Incorrect global step averaging would produce a different result.
    incorrect = (
        2 * math.log(2) + 3 * math.log(4)
    ) / 5

    assert abs(actual.item() - incorrect) > 0.01


def test_combined_formula_and_lambda_zero(example):
    logits, rules, demos, candidates, targets = example

    final = final_only_loss(
        logits, rules, demos, candidates
    )
    auxiliary = revision_loss(
        logits, targets, demos, candidates
    )

    torch.testing.assert_close(
        revision_aware_loss(*example, lambda_revision=0),
        final,
    )

    torch.testing.assert_close(
        revision_aware_loss(*example, lambda_revision=0.75),
        final + 0.75 * auxiliary,
    )


def test_mixed_length_final_step_selection(example):
    logits, rules, demos, candidates, _ = example
    logits = logits.clone()

    # Episode 0 ends at step 1, not step 2.
    logits[0, 0] = torch.tensor([0., 12., 0., 0.])
    logits[0, 1] = torch.tensor([12., 0., 0., 0.])

    # Episode 1 ends at step 2.
    logits[1, 0] = torch.tensor([0., 0., 12., 0.])
    logits[1, 2] = torch.tensor([0., 12., 0., 0.])

    final_rows = torch.stack((
        logits[0, 1].masked_fill(
            ~candidates[0], float("-inf")
        ),
        logits[1, 2],
    ))

    expected = F.cross_entropy(final_rows, rules)

    torch.testing.assert_close(
        final_only_loss(logits, rules, demos, candidates),
        expected,
    )

    assert expected.item() > 6.0


def test_known_soft_target_cross_entropy():
    logits = torch.tensor(
        [[[math.log(3), 0., 1e10]]],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [[[0.25, 0.75, 0.]]],
        dtype=torch.float64,
    )
    demos = torch.ones(1, 1, dtype=torch.bool)
    candidates = torch.tensor(
        [[1, 1, 0]], dtype=torch.bool
    )

    # Real candidate probabilities are 0.75 and 0.25.
    expected = -(
        0.25 * math.log(0.75)
        + 0.75 * math.log(0.25)
    )

    torch.testing.assert_close(
        revision_loss(logits, targets, demos, candidates),
        torch.tensor(expected, dtype=logits.dtype),
    )


def test_padded_candidates_do_not_affect_losses(example):
    logits, rules, demos, candidates, targets = example

    logits2 = logits.clone()
    targets2 = targets.clone()

    invalid = (~candidates[:, None, :]).expand_as(logits)

    logits2[invalid] = 1e20
    targets2[invalid] = 123.0

    torch.testing.assert_close(
        final_only_loss(logits2, rules, demos, candidates),
        final_only_loss(logits, rules, demos, candidates),
    )

    torch.testing.assert_close(
        revision_aware_loss(
            logits2, rules, demos, candidates, targets2
        ),
        revision_aware_loss(*example),
    )


def test_padded_demonstration_values_do_not_affect_losses(example):
    logits, rules, demos, candidates, targets = example

    logits2 = logits.clone()
    targets2 = targets.clone()

    logits2[~demos] = float("nan")
    targets2[~demos] = float("nan")

    torch.testing.assert_close(
        final_only_loss(logits2, rules, demos, candidates),
        final_only_loss(logits, rules, demos, candidates),
    )

    torch.testing.assert_close(
        revision_aware_loss(
            logits2, rules, demos, candidates, targets2
        ),
        revision_aware_loss(*example),
    )


def test_large_logits_numerically_stable_and_finite_gradients(example):
    logits, rules, demos, candidates, targets = example

    logits = (
        logits + torch.tensor(
            [
                [[1e4, -1e4, 0, 0]] * 3,
                [[1e4, -1e4, 1e3, -1e3]] * 3,
            ],
            dtype=logits.dtype,
        )
    ).requires_grad_()

    loss = revision_aware_loss(
        logits,
        rules,
        demos,
        candidates,
        targets,
        lambda_revision=0.5,
    )

    assert torch.isfinite(loss)

    loss.backward()

    assert torch.isfinite(logits.grad).all()

    invalid = ~(
        demos.unsqueeze(-1) & candidates.unsqueeze(1)
    )

    assert torch.all(logits.grad[invalid] == 0)
    assert torch.any(logits.grad[~invalid] != 0)


def test_baseline_gradients_only_from_final_steps(example):
    logits, rules, demos, candidates, _ = example

    logits = logits.clone().requires_grad_()

    final_only_loss(
        logits, rules, demos, candidates
    ).backward()

    assert (logits.grad[0, 0] == 0).all()
    assert (logits.grad[0, 2] == 0).all()
    assert (logits.grad[1, :2] == 0).all()

    assert logits.grad[0, 1, 1] != 0
    assert logits.grad[1, 2, 2] != 0


def test_lambda_zero_matches_baseline_gradients(example):
    logits, rules, demos, candidates, targets = example

    a = logits.clone().requires_grad_()
    b = logits.clone().requires_grad_()

    baseline = final_only_loss(
        a, rules, demos, candidates
    )
    combined = revision_aware_loss(
        b,
        rules,
        demos,
        candidates,
        targets,
        lambda_revision=0,
    )

    torch.testing.assert_close(
        baseline, combined, rtol=0, atol=0
    )

    torch.testing.assert_close(
        torch.autograd.grad(baseline, a)[0],
        torch.autograd.grad(combined, b)[0],
        rtol=0,
        atol=0,
    )


def test_invalid_rule_index_rejected(example):
    logits, rules, demos, candidates, _ = example

    with pytest.raises(ValueError, match="padded candidate"):
        final_only_loss(
            logits,
            torch.tensor([3, 2]),
            demos,
            candidates,
        )


def test_invalid_target_mass_rejected(example):
    logits, _, demos, candidates, targets = example

    targets = targets.clone()
    targets[0, 1, :2] = 0.1

    with pytest.raises(ValueError, match="sum to one"):
        revision_loss(
            logits, targets, demos, candidates
        )


@pytest.mark.parametrize(
    "coefficient",
    [-1, float("nan"), float("inf")],
)
def test_invalid_lambda_rejected(example, coefficient):
    with pytest.raises(ValueError, match="lambda_revision"):
        revision_aware_loss(
            *example, lambda_revision=coefficient
        )


def test_empty_sequence_rejected(example):
    logits, rules, demos, candidates, _ = example

    demos = demos.clone()
    demos[0] = False

    with pytest.raises(ValueError, match="nonempty prefix"):
        final_only_loss(
            logits, rules, demos, candidates
        )


def test_nonprefix_demo_mask_rejected(example):
    logits, rules, demos, candidates, _ = example

    demos = demos.clone()
    demos[0] = torch.tensor([True, False, True])

    with pytest.raises(ValueError, match="nonempty prefix"):
        final_only_loss(
            logits, rules, demos, candidates
        )


def test_invalid_candidate_logits_ignored_even_if_nan(example):
    logits, rules, demos, candidates, targets = example

    modified = logits.clone()
    invalid = ~candidates[:, None, :].expand_as(logits)

    modified[invalid] = float("nan")

    loss = revision_aware_loss(
        modified,
        rules,
        demos,
        candidates,
        targets,
    )

    assert torch.isfinite(loss)


def test_real_dataset_and_gru_integration():
    """Test A05 -> A07 -> A08 without requiring trained weights."""
    torch.manual_seed(9)

    dataset = ARCEpisodeDataset([
        generate_episode(
            seed=101, num_demos=2, ambiguous=True
        ),
        generate_episode(
            seed=202, num_demos=5, ambiguous=True
        ),
    ])
    batch = collate_episodes([dataset[0], dataset[1]])

    model = SequentialGRUHypothesisClassifier(
        demonstration_dim=24,
        hidden_dim=32,
        program_dim=20,
    )

    logits = model(
        batch.demonstrations,
        batch.demo_mask,
        batch.op_ids,
        batch.op_params,
        batch.op_mask,
        batch.param_mask,
        batch.candidate_mask,
    )

    assert logits.shape == (2, 5, 64)
    assert batch.num_demos.tolist() == [2, 5]

    final = final_only_loss(
        logits,
        batch.rule_index,
        batch.demo_mask,
        batch.candidate_mask,
    )

    total = revision_aware_loss(
        logits,
        batch.rule_index,
        batch.demo_mask,
        batch.candidate_mask,
        batch.soft_targets,
        lambda_revision=0.5,
    )

    assert torch.isfinite(final)
    assert torch.isfinite(total)

    total.backward()

    grad = model.recurrent.weight_ih_l0.grad

    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.any(grad != 0)