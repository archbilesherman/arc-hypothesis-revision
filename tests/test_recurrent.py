import pytest
import torch

from arc_revision.data.dataset import (
    ARCEpisodeDataset,
    collate_episodes,
)
from arc_revision.data.encoding import OP_IDS
from arc_revision.data.generator import generate_episode
from arc_revision.models.recurrent import (
    SequentialGRUHypothesisClassifier,
)


@pytest.fixture(scope="module")
def episodes():
    """Real A05 samples with 2 and 5 demonstrations."""
    dataset = ARCEpisodeDataset([
        generate_episode(
            seed=101, num_demos=2, ambiguous=True
        ),
        generate_episode(
            seed=202, num_demos=5, ambiguous=True
        ),
    ])

    return dataset[0], dataset[1]


@pytest.fixture(scope="module")
def mixed_batch(episodes):
    return collate_episodes(list(episodes))


def _model():
    torch.manual_seed(123)
    return SequentialGRUHypothesisClassifier(
        demonstration_dim=24,
        hidden_dim=32,
        program_dim=20,
    )


def _inputs(batch):
    return (
        batch.demonstrations,
        batch.demo_mask,
        batch.op_ids,
        batch.op_params,
        batch.op_mask,
        batch.param_mask,
        batch.candidate_mask,
    )


def test_real_batch_output_shape_and_variable_lengths(
    mixed_batch,
):
    model = _model()
    logits = model(*_inputs(mixed_batch))

    assert mixed_batch.num_demos.tolist() == [2, 5]
    assert mixed_batch.num_candidates.tolist() == [8, 64]
    assert logits.shape == (2, 5, 64)
    assert model.recurrent.bidirectional is False


def test_valid_and_padded_logits(mixed_batch):
    logits = _model()(*_inputs(mixed_batch))

    valid = (
        mixed_batch.demo_mask[:, :, None]
        & mixed_batch.candidate_mask[:, None, :]
    )

    invalid_candidates = (
        mixed_batch.demo_mask[:, :, None]
        & ~mixed_batch.candidate_mask[:, None, :]
    ).expand_as(logits)

    assert torch.isfinite(logits[valid]).all()
    assert torch.isneginf(logits[invalid_candidates]).all()
    assert torch.all(
        logits[~mixed_batch.demo_mask] == 0
    )


def test_candidate_permutation_reorders_logits(mixed_batch):
    model = _model().eval()

    permutation = torch.arange(
        mixed_batch.op_ids.shape[1] - 1, -1, -1
    )

    with torch.no_grad():
        original = model(*_inputs(mixed_batch))

        reordered = model(
            mixed_batch.demonstrations,
            mixed_batch.demo_mask,
            mixed_batch.op_ids[:, permutation],
            mixed_batch.op_params[:, permutation],
            mixed_batch.op_mask[:, permutation],
            mixed_batch.param_mask[:, permutation],
            mixed_batch.candidate_mask[:, permutation],
        )

    torch.testing.assert_close(
        reordered, original[:, :, permutation]
    )


def test_future_demonstrations_cannot_change_earlier_logits(
    mixed_batch,
):
    model = _model().eval()

    changed = mixed_batch.demonstrations.clone()
    changed[1, 4] = (changed[1, 4] + 1) % 10

    with torch.no_grad():
        original = model(*_inputs(mixed_batch))

        updated = model(
            changed,
            mixed_batch.demo_mask,
            mixed_batch.op_ids,
            mixed_batch.op_params,
            mixed_batch.op_mask,
            mixed_batch.param_mask,
            mixed_batch.candidate_mask,
        )

    torch.testing.assert_close(
        updated[1, :4], original[1, :4]
    )

    assert not torch.allclose(
        updated[1, 4], original[1, 4]
    )


def test_padding_values_do_not_change_predictions(mixed_batch):
    model = _model().eval()

    demos = mixed_batch.demonstrations.clone()
    params = mixed_batch.op_params.clone()
    ids = mixed_batch.op_ids.clone()

    demos[~mixed_batch.demo_mask] = -1
    params[~mixed_batch.param_mask] = 999
    ids[~mixed_batch.op_mask] = OP_IDS["reflect_horizontal"]

    with torch.no_grad():
        original = model(*_inputs(mixed_batch))

        changed = model(
            demos,
            mixed_batch.demo_mask,
            ids,
            params,
            mixed_batch.op_mask,
            mixed_batch.param_mask,
            mixed_batch.candidate_mask,
        )

    torch.testing.assert_close(changed, original)


def test_episode_scores_independent_of_other_batch_members(
    episodes, mixed_batch,
):
    model = _model().eval()

    single_batch = collate_episodes([episodes[0]])

    with torch.no_grad():
        single = model(*_inputs(single_batch))
        mixed = model(*_inputs(mixed_batch))

    torch.testing.assert_close(
        mixed[0, :2, :8],
        single[0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_hidden_state_does_not_leak_across_calls(episodes):
    model = _model().eval()

    first_batch = collate_episodes([episodes[0]])
    other_batch = collate_episodes([episodes[1]])

    with torch.no_grad():
        before = model(*_inputs(first_batch))
        _ = model(*_inputs(other_batch))
        after = model(*_inputs(first_batch))

    torch.testing.assert_close(after, before)


def test_gradients_reach_all_active_model_components(
    mixed_batch,
):
    model = _model()
    logits = model(*_inputs(mixed_batch))

    valid = (
        mixed_batch.demo_mask[:, :, None]
        & mixed_batch.candidate_mask[:, None, :]
    )

    loss = logits[valid].square().mean()
    loss.backward()

    parameters = [
        model.demonstration_encoder.grid_encoder[0].weight,
        model.demonstration_encoder.pair_projection.weight,
        model.recurrent.weight_ih_l0,
        model.program_encoder.operation_embedding.weight,
        model.program_encoder.parameter_embedding.weight,
        model.program_encoder.operation_gru.weight_ih_l0,
        model.task_projection.weight,
    ]

    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.any(parameter.grad != 0)


def test_invalid_demonstration_mask_shape_is_rejected(
    mixed_batch,
):
    model = _model()
    args = list(_inputs(mixed_batch))

    args[1] = args[1][:, :-1]

    with pytest.raises(ValueError, match="demo_mask"):
        model(*args)


def test_nonprefix_demonstration_mask_is_rejected(
    mixed_batch,
):
    model = _model()
    args = list(_inputs(mixed_batch))

    args[1] = args[1].clone()
    args[1][0] = torch.tensor(
        [True, False, True, False, False]
    )

    with pytest.raises(ValueError, match="prefix"):
        model(*args)


def test_empty_demonstration_sequence_is_rejected(
    mixed_batch,
):
    model = _model()
    args = list(_inputs(mixed_batch))

    args[1] = args[1].clone()
    args[1][0] = False

    with pytest.raises(
        ValueError, match="at least one demonstration"
    ):
        model(*args)


def test_empty_candidate_set_is_rejected(mixed_batch):
    model = _model()
    args = list(_inputs(mixed_batch))

    args[6] = args[6].clone()
    args[6][0] = False

    with pytest.raises(
        ValueError, match="at least one candidate"
    ):
        model(*args)