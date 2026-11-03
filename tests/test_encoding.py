"""Tests for the deterministic DSL program encoding."""

import json
import random
from typing import get_args

import pytest
import torch

import arc_revision.data.encoding as encoding
from arc_revision.data.encoding import (
    NUM_OP_IDS,
    NUM_PARAM_VALUES,
    OP_ARITY,
    OP_IDS,
    OP_KINDS,
    PAD_OP_ID,
    PAD_PARAM_VALUE,
    PARAM_CAPACITY,
    decode_hypotheses,
    encode_hypotheses,
    encoding_metadata,
    operation_masks,
)
from arc_revision.data.generator import generate_episode
from arc_revision.data.hypotheses import Hypothesis, OpKind, Operation


# One or more valid operations of every kind, covering every arity.
EXAMPLE_OPERATIONS = (
    Operation("rotate", (90,)),
    Operation("rotate", (180,)),
    Operation("rotate", (270,)),
    Operation("reflect_horizontal"),
    Operation("reflect_vertical"),
    Operation("reflect_main_diagonal"),
    Operation("reflect_anti_diagonal"),
    Operation("replace_color", (0, 9)),
    Operation("replace_color", (4, 1)),
    Operation("swap_colors", (2, 5)),
    Operation("permute_colors", (3, 1, 4, 0, 5, 9, 2, 6, 8, 7)),
)


def random_space(seed: int, size: int = 12) -> tuple[Hypothesis, ...]:
    """Distinct random programs of length 0-6 using every kind."""

    rng = random.Random(seed)
    programs = [
        Hypothesis(tuple(
            rng.choice(EXAMPLE_OPERATIONS)
            for _ in range(rng.randint(0, 6))
        ))
        for _ in range(size)
    ]

    return tuple(dict.fromkeys(programs))


def test_vocabulary_covers_dsl():
    assert set(OP_KINDS) == set(get_args(OpKind))
    assert len(set(OP_KINDS)) == len(OP_KINDS)
    assert set(OP_ARITY) == set(OP_KINDS)
    assert PAD_OP_ID not in OP_IDS.values()
    assert NUM_OP_IDS == len(OP_KINDS) + 1


def test_operation_ids_are_stable():
    """IDs are part of saved checkpoints; changes need a new version."""

    assert OP_IDS == {
        "rotate": 1,
        "reflect_horizontal": 2,
        "reflect_vertical": 3,
        "reflect_main_diagonal": 4,
        "reflect_anti_diagonal": 5,
        "replace_color": 6,
        "swap_colors": 7,
        "permute_colors": 8,
    }


def test_arity_matches_operation_validation():
    assert {op.kind for op in EXAMPLE_OPERATIONS} == set(OP_KINDS)

    for op in EXAMPLE_OPERATIONS:
        assert len(op.params) == OP_ARITY[op.kind]

    assert PARAM_CAPACITY == 10


def test_known_encoding():
    encoded = encode_hypotheses((
        Hypothesis((
            Operation("rotate", (270,)),
            Operation("replace_color", (3, 7)),
        )),
        Hypothesis((Operation("reflect_vertical"),)),
        Hypothesis(),
    ))

    expected_params = torch.zeros((3, 2, PARAM_CAPACITY), dtype=torch.long)
    expected_params[0, 0, 0] = 3
    expected_params[0, 1, :2] = torch.tensor([3, 7])

    assert encoded.op_ids.dtype == torch.long
    assert encoded.op_params.dtype == torch.long
    assert encoded.op_ids.tolist() == [[1, 6], [3, 0], [0, 0]]
    assert torch.equal(encoded.op_params, expected_params)

    assert encoded.op_mask.dtype == torch.bool
    assert encoded.op_mask.tolist() == [
        [True, True],
        [True, False],
        [False, False],
    ]

    assert encoded.param_mask.shape == (3, 2, PARAM_CAPACITY)
    assert encoded.param_mask[0, 0].tolist() == [True] + [False] * 9
    assert encoded.param_mask[0, 1].tolist() == [True] * 2 + [False] * 8
    assert not encoded.param_mask[1:].any()


@pytest.mark.parametrize("degrees,turns", [(90, 1), (180, 2), (270, 3)])
def test_rotation_is_stored_as_quarter_turns(degrees, turns):
    program = Hypothesis((Operation("rotate", (degrees,)),))
    encoded = encode_hypotheses((program,))

    assert encoded.op_params[0, 0, 0].item() == turns
    assert decode_hypotheses(
        encoded.op_ids, encoded.op_params
    ) == (program,)


def test_permutation_keeps_all_parameters():
    op = Operation("permute_colors", (3, 1, 4, 0, 5, 9, 2, 6, 8, 7))
    encoded = encode_hypotheses((Hypothesis((op,)),))

    assert encoded.op_params[0, 0].tolist() == list(op.params)
    assert encoded.param_mask[0, 0].all()


def test_execution_order_is_preserved():
    first = Operation("rotate", (90,))
    second = Operation("reflect_horizontal")
    forward = Hypothesis((first, second))
    backward = Hypothesis((second, first))

    encoded = encode_hypotheses((forward, backward))

    assert encoded.op_ids.tolist() == [
        [OP_IDS["rotate"], OP_IDS["reflect_horizontal"]],
        [OP_IDS["reflect_horizontal"], OP_IDS["rotate"]],
    ]
    assert decode_hypotheses(
        encoded.op_ids, encoded.op_params
    ) == (forward, backward)


def test_candidate_order_is_preserved():
    space = random_space(seed=3)
    order = list(reversed(range(len(space))))
    reordered = tuple(space[i] for i in order)

    original = encode_hypotheses(space)
    permuted = encode_hypotheses(reordered)

    # Rows follow candidate order, padded to the same longest program.
    assert torch.equal(permuted.op_ids, original.op_ids[order])
    assert torch.equal(permuted.op_params, original.op_params[order])


def test_encoding_is_deterministic():
    space = random_space(seed=5)
    rebuilt = tuple(
        Hypothesis(tuple(
            Operation(op.kind, tuple(op.params))
            for op in h.operations
        ))
        for h in space
    )

    first = encode_hypotheses(space)
    second = encode_hypotheses(rebuilt)

    assert torch.equal(first.op_ids, second.op_ids)
    assert torch.equal(first.op_params, second.op_params)


@pytest.mark.parametrize("seed", range(20))
def test_round_trip_random_programs(seed):
    space = random_space(seed)
    encoded = encode_hypotheses(space)

    assert decode_hypotheses(encoded.op_ids, encoded.op_params) == space


@pytest.mark.parametrize("num_demos", [2, 3, 4, 5])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_generated_candidate_spaces(num_demos, ambiguous):
    episode = generate_episode(
        seed=num_demos,
        num_demos=num_demos,
        ambiguous=ambiguous,
    )

    encoded = encode_hypotheses(episode.hypotheses)
    num_candidates = len(episode.hypotheses)
    lengths = [len(h.operations) for h in episode.hypotheses]

    assert encoded.op_ids.shape == (num_candidates, max(lengths))
    assert encoded.op_params.shape == (
        num_candidates, max(lengths), PARAM_CAPACITY
    )
    assert encoded.op_mask.sum(dim=-1).tolist() == lengths
    assert decode_hypotheses(
        encoded.op_ids, encoded.op_params
    ) == episode.hypotheses


def test_padding_and_value_ranges():
    encoded = encode_hypotheses(random_space(seed=11))
    op_mask, param_mask = encoded.op_mask, encoded.param_mask

    assert (encoded.op_ids[~op_mask] == PAD_OP_ID).all()
    assert (encoded.op_params[~param_mask] == PAD_PARAM_VALUE).all()
    assert not param_mask[~op_mask].any()

    assert (encoded.op_ids >= 0).all()
    assert (encoded.op_ids < NUM_OP_IDS).all()
    assert (encoded.op_params >= 0).all()
    assert (encoded.op_params < NUM_PARAM_VALUES).all()


def test_empty_programs_keep_one_padded_column():
    encoded = encode_hypotheses((Hypothesis(),))

    assert encoded.op_ids.tolist() == [[PAD_OP_ID]]
    assert encoded.op_params.shape == (1, 1, PARAM_CAPACITY)
    assert not encoded.op_mask.any()
    assert decode_hypotheses(
        encoded.op_ids, encoded.op_params
    ) == (Hypothesis(),)


@pytest.mark.parametrize("hypotheses", [
    (),
    [Hypothesis()],
    (Hypothesis(), Hypothesis()),
])
def test_rejects_invalid_hypothesis_space(hypotheses):
    with pytest.raises(ValueError):
        encode_hypotheses(hypotheses)


def test_rejects_operation_without_encoding(monkeypatch):
    """A DSL extension must update the encoding instead of failing silently."""

    monkeypatch.delitem(encoding.OP_IDS, "swap_colors")

    with pytest.raises(ValueError, match="No encoding"):
        encode_hypotheses((
            Hypothesis((Operation("swap_colors", (1, 2)),)),
        ))


def test_operation_masks_support_batched_ids():
    op_ids = torch.tensor([
        [[OP_IDS["rotate"], OP_IDS["replace_color"], PAD_OP_ID]],
        [[OP_IDS["reflect_vertical"], PAD_OP_ID, PAD_OP_ID]],
    ])

    op_mask, param_mask = operation_masks(op_ids)

    assert op_mask.shape == (2, 1, 3)
    assert param_mask.shape == (2, 1, 3, PARAM_CAPACITY)
    assert param_mask.sum(dim=-1).tolist() == [[[1, 2, 0]], [[0, 0, 0]]]


@pytest.mark.parametrize("op_ids,error", [
    ([[1]], TypeError),
    (torch.tensor([[1]], dtype=torch.int32), TypeError),
    (torch.tensor(1), ValueError),
    (torch.tensor([[NUM_OP_IDS]]), ValueError),
    (torch.tensor([[-1]]), ValueError),
])
def test_operation_masks_reject_invalid_ids(op_ids, error):
    with pytest.raises(error):
        operation_masks(op_ids)


def _corrupt(name):
    """Return a valid encoding of one program, then corrupt it."""

    encoded = encode_hypotheses((
        Hypothesis((
            Operation("rotate", (90,)),
            Operation("replace_color", (3, 7)),
        )),
        Hypothesis((Operation("reflect_vertical"),)),
    ))
    op_ids = encoded.op_ids.clone()
    op_params = encoded.op_params.clone()

    if name == "gap":
        op_ids[1] = torch.tensor([PAD_OP_ID, OP_IDS["reflect_vertical"]])
    elif name == "padded_param":
        op_params[1, 0, 0] = 5
    elif name == "padded_slot_param":
        op_params[1, 1, 3] = 1
    elif name == "bad_rotation":
        op_params[0, 0, 0] = 4
    elif name == "equal_colors":
        op_params[0, 1, :2] = torch.tensor([3, 3])
    elif name == "unknown_id":
        op_ids[0, 0] = NUM_OP_IDS
    elif name == "wrong_capacity":
        op_params = op_params[..., :-1]
    elif name == "wrong_rank":
        op_ids = op_ids[0]
        op_params = op_params[0]

    return op_ids, op_params


@pytest.mark.parametrize("name,match", [
    ("gap", "precede"),
    ("padded_param", "Unused"),
    ("padded_slot_param", "Unused"),
    ("bad_rotation", "rotate"),
    ("equal_colors", "differ"),
    ("unknown_id", "op_ids"),
    ("wrong_capacity", "shape"),
    ("wrong_rank", "shape"),
])
def test_decode_rejects_corrupted_encodings(name, match):
    op_ids, op_params = _corrupt(name)

    with pytest.raises(ValueError, match=match):
        decode_hypotheses(op_ids, op_params)


def test_decode_rejects_wrong_dtype():
    encoded = encode_hypotheses((Hypothesis(),))

    with pytest.raises(TypeError):
        decode_hypotheses(encoded.op_ids, encoded.op_params.int())


def test_metadata_is_json_serializable():
    metadata = encoding_metadata()

    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["op_kinds"] == list(OP_KINDS)
    assert metadata["param_capacity"] == PARAM_CAPACITY
    assert type(metadata["version"]) is int
