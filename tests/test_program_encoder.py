import pytest
import torch

from arc_revision.data.encoding import (
    NUM_OP_IDS,
    NUM_PARAM_VALUES,
    OP_IDS,
    PARAM_CAPACITY,
    operation_masks,
)
from arc_revision.models.program_encoder import ProgramEncoder


def _program_batch():
    """Two episodes with mixed candidate and operation lengths."""
    ids = torch.tensor([
        [
            [0, 0, 0],
            [OP_IDS["replace_color"], 0, 0],
            [OP_IDS["rotate"], OP_IDS["reflect_horizontal"], 0],
        ],
        [
            [OP_IDS["rotate"], 0, 0],
            [OP_IDS["reflect_vertical"], OP_IDS["rotate"], 0],
            [0, 0, 0],
        ],
    ], dtype=torch.long)

    params = torch.zeros(
        (*ids.shape, PARAM_CAPACITY), dtype=torch.long
    )
    params[0, 1, 0, :2] = torch.tensor([1, 2])
    params[0, 2, 0, 0] = 1
    params[1, 0, 0, 0] = 2
    params[1, 1, 1, 0] = 3

    op_mask, param_mask = operation_masks(ids)
    candidate_mask = torch.tensor([
        [True, True, True],
        [True, True, False],
    ])

    return ids, params, op_mask, param_mask, candidate_mask


def test_output_shape_and_finite_values():
    torch.manual_seed(7)
    model = ProgramEncoder(embedding_dim=32)
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    output = model(
        ids, params, op_mask, param_mask, candidate_mask
    )

    assert output.shape == (2, 3, 32)
    assert torch.isfinite(output).all()


def test_identity_program_uses_learned_embedding():
    model = ProgramEncoder(embedding_dim=24)
    output = model(*_program_batch())

    torch.testing.assert_close(
        output[0, 0], model.identity_embedding
    )
    assert torch.all(output[1, 2] == 0)


def test_parameter_positions_affect_representation():
    torch.manual_seed(11)
    model = ProgramEncoder(embedding_dim=32)

    ids = torch.full(
        (1, 2, 1), OP_IDS["replace_color"], dtype=torch.long
    )
    params = torch.zeros(
        (1, 2, 1, PARAM_CAPACITY), dtype=torch.long
    )
    params[0, 0, 0, :2] = torch.tensor([1, 2])
    params[0, 1, 0, :2] = torch.tensor([2, 1])

    op_mask, param_mask = operation_masks(ids)

    output = model(
        ids,
        params,
        op_mask,
        param_mask,
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert not torch.allclose(output[0, 0], output[0, 1])


def test_operation_order_affects_representation():
    torch.manual_seed(12)
    model = ProgramEncoder(embedding_dim=32)

    rotate = OP_IDS["rotate"]
    reflect = OP_IDS["reflect_horizontal"]

    ids = torch.tensor([
        [[rotate, reflect], [reflect, rotate]]
    ])

    params = torch.zeros(
        (1, 2, 2, PARAM_CAPACITY), dtype=torch.long
    )
    params[0, 0, 0, 0] = 1
    params[0, 1, 1, 0] = 1

    op_mask, param_mask = operation_masks(ids)

    output = model(
        ids,
        params,
        op_mask,
        param_mask,
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert not torch.allclose(output[0, 0], output[0, 1])


def test_masked_operation_and_parameter_values_do_not_matter():
    model = ProgramEncoder(embedding_dim=32).eval()
    original = _program_batch()

    ids, params, op_mask, param_mask, candidate_mask = [
        x.clone() for x in original
    ]

    ids[~op_mask] = OP_IDS["reflect_vertical"]
    params[~param_mask] = 999

    with torch.no_grad():
        expected = model(*original)
        actual = model(
            ids, params, op_mask, param_mask, candidate_mask
        )

    torch.testing.assert_close(actual, expected)


def test_invalid_candidates_are_zero():
    model = ProgramEncoder(embedding_dim=16)
    *_, candidate_mask = _program_batch()

    output = model(*_program_batch())

    assert torch.all(output[~candidate_mask] == 0)


def test_gradients_reach_all_program_components():
    model = ProgramEncoder(embedding_dim=24)
    *inputs, candidate_mask = _program_batch()

    output = model(*inputs, candidate_mask)
    loss = output[candidate_mask].square().mean()
    loss.backward()

    components = [
        model.operation_embedding.weight,
        model.parameter_embedding.weight,
        model.operation_gru.weight_ih_l0,
        model.identity_embedding,
    ]

    for parameter in components:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.any(parameter.grad != 0)


def test_invalid_parameter_shape_is_rejected():
    model = ProgramEncoder()
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    with pytest.raises(ValueError, match="op_params"):
        model(
            ids,
            params[..., :-1],
            op_mask,
            param_mask,
            candidate_mask,
        )


def test_invalid_candidate_mask_shape_is_rejected():
    model = ProgramEncoder()
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    with pytest.raises(ValueError, match="candidate_mask"):
        model(
            ids,
            params,
            op_mask,
            param_mask,
            candidate_mask[:, :2],
        )


def test_nonprefix_operation_mask_is_rejected():
    model = ProgramEncoder()
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    op_mask[0, 1] = torch.tensor(
        [False, True, False]
    )

    with pytest.raises(ValueError, match="prefix"):
        model(
            ids, params, op_mask, param_mask, candidate_mask
        )


def test_invalid_operation_id_is_rejected():
    model = ProgramEncoder()
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    ids[0, 1, 0] = NUM_OP_IDS

    with pytest.raises(ValueError, match="Operation ID"):
        model(
            ids, params, op_mask, param_mask, candidate_mask
        )


def test_invalid_valid_parameter_is_rejected():
    model = ProgramEncoder()
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    params[0, 1, 0, 0] = NUM_PARAM_VALUES

    with pytest.raises(ValueError, match="Parameter value"):
        model(
            ids, params, op_mask, param_mask, candidate_mask
        )


def test_parameter_mask_cannot_extend_padded_operation():
    model = ProgramEncoder()
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    param_mask[0, 1, 2, 0] = True

    with pytest.raises(ValueError, match="Padded operations"):
        model(
            ids, params, op_mask, param_mask, candidate_mask
        )


def test_invalid_mask_dtype_is_rejected():
    model = ProgramEncoder()
    ids, params, op_mask, param_mask, candidate_mask = (
        _program_batch()
    )

    with pytest.raises(TypeError, match="boolean"):
        model(
            ids,
            params,
            op_mask.float(),
            param_mask,
            candidate_mask,
        )