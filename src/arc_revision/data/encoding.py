"""Deterministic, order-preserving tensor encoding of DSL programs.

An ordered set of K candidate programs is encoded as

    op_ids:    [K, L]    torch.long   PAD_OP_ID after each program ends
    op_params: [K, L, P] torch.long   PAD_PARAM_VALUE in unused slots

Row k encodes hypotheses[k] and column l its l-th operation, so both
candidate order and execution order are preserved. L is the longest
program in the set (at least 1, so an empty identity program still
occupies one fully padded row) and P = PARAM_CAPACITY.

Operation IDs start at 1 so that 0 can mark padding. Parameters keep
their positional order; rotations are stored as clockwise quarter
turns (90 -> 1, 180 -> 2, 270 -> 3), so every encoded parameter lies
in [0, NUM_PARAM_VALUES). Colour 0 is a real parameter value, so
validity must come from operation_masks, never from the stored value.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from arc_revision.data.hypotheses import (
    NUM_COLORS,
    Hypothesis,
    OpKind,
    Operation,
    validate_hypothesis_space,
)


# Bump when any constant below changes meaning; checkpoints record it.
ENCODING_VERSION = 1

# Stable ordering: an operation's ID is its position here plus one.
OP_KINDS: tuple[OpKind, ...] = (
    "rotate",
    "reflect_horizontal",
    "reflect_vertical",
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
    "replace_color",
    "swap_colors",
    "permute_colors",
)

PAD_OP_ID = 0
OP_IDS: dict[str, int] = {
    kind: index + 1 for index, kind in enumerate(OP_KINDS)
}
# Vocabulary size for an operation embedding, padding included.
NUM_OP_IDS = len(OP_KINDS) + 1

# Parameter count of each operation, mirroring Operation validation.
OP_ARITY: dict[str, int] = {
    "rotate": 1,
    "reflect_horizontal": 0,
    "reflect_vertical": 0,
    "reflect_main_diagonal": 0,
    "reflect_anti_diagonal": 0,
    "replace_color": 2,
    "swap_colors": 2,
    "permute_colors": NUM_COLORS,
}

PARAM_CAPACITY = max(OP_ARITY.values())
PAD_PARAM_VALUE = 0
# Encoded parameters are colours (0-9) or quarter turns (1-3).
NUM_PARAM_VALUES = NUM_COLORS
ROTATION_UNIT_DEGREES = 90

# Arity indexed by operation ID; padding has no parameters.
_ARITY_BY_ID = torch.tensor(
    [0] + [OP_ARITY[kind] for kind in OP_KINDS],
    dtype=torch.long,
)


@dataclass(frozen=True)
class EncodedPrograms:
    """
    Padded encoding of an ordered candidate program set.

    Shapes:
        op_ids:    [K, L]    torch.long
        op_params: [K, L, P] torch.long
    """

    op_ids: Tensor
    op_params: Tensor

    @property
    def op_mask(self) -> Tensor:
        """[K, L] bool: True where a real operation is stored."""
        return operation_masks(self.op_ids)[0]

    @property
    def param_mask(self) -> Tensor:
        """[K, L, P] bool: True where a real parameter is stored."""
        return operation_masks(self.op_ids)[1]


def operation_masks(op_ids: Tensor) -> tuple[Tensor, Tensor]:
    """
    Derive validity masks from operation IDs of any leading shape.

    Args:
        op_ids: [..., L] torch.long operation IDs.

    Returns:
        op_mask [..., L] and param_mask [..., L, P], both torch.bool.
        A parameter slot is valid when its operation is real and the
        slot index is below that operation's arity.
    """

    if not isinstance(op_ids, Tensor):
        raise TypeError("op_ids must be a PyTorch tensor")

    if op_ids.dtype != torch.long:
        raise TypeError("op_ids must have dtype torch.long")

    if op_ids.ndim == 0:
        raise ValueError("op_ids must have at least one dimension")

    if ((op_ids < 0) | (op_ids >= NUM_OP_IDS)).any().item():
        raise ValueError(
            f"op_ids must be between 0 and {NUM_OP_IDS - 1}"
        )

    arity = _ARITY_BY_ID.to(op_ids.device)[op_ids]
    slots = torch.arange(PARAM_CAPACITY, device=op_ids.device)

    return op_ids != PAD_OP_ID, slots < arity.unsqueeze(-1)


def _encode_operation(op: Operation) -> tuple[int, tuple[int, ...]]:
    """Map one operation to its ID and encoded parameters."""

    if op.kind not in OP_IDS:
        raise ValueError(
            f"No encoding for operation {op.kind!r}; extend OP_KINDS "
            "and OP_ARITY and bump ENCODING_VERSION"
        )

    params = op.params

    if op.kind == "rotate":
        params = (params[0] // ROTATION_UNIT_DEGREES,)

    if len(params) != OP_ARITY[op.kind]:
        raise ValueError(
            f"{op.kind} has {len(params)} parameters but "
            f"OP_ARITY expects {OP_ARITY[op.kind]}"
        )

    return OP_IDS[op.kind], params


def encode_hypotheses(
    hypotheses: tuple[Hypothesis, ...],
) -> EncodedPrograms:
    """
    Encode candidate programs, preserving candidate and operation order.

    The result depends only on the programs themselves, so equal
    hypothesis tuples always produce identical tensors.

    Returns:
        EncodedPrograms with op_ids [K, L] and op_params [K, L, P],
        where L = max(1, longest program length).
    """

    validate_hypothesis_space(hypotheses)

    length = max(
        1,
        max(len(h.operations) for h in hypotheses),
    )

    op_ids = [[PAD_OP_ID] * length for _ in hypotheses]
    op_params = [
        [[PAD_PARAM_VALUE] * PARAM_CAPACITY for _ in range(length)]
        for _ in hypotheses
    ]

    for row, hypothesis in enumerate(hypotheses):
        for position, op in enumerate(hypothesis.operations):
            op_id, params = _encode_operation(op)
            op_ids[row][position] = op_id
            op_params[row][position][:len(params)] = params

    return EncodedPrograms(
        op_ids=torch.tensor(op_ids, dtype=torch.long),
        op_params=torch.tensor(op_params, dtype=torch.long),
    )


def decode_hypotheses(
    op_ids: Tensor,
    op_params: Tensor,
) -> tuple[Hypothesis, ...]:
    """
    Invert encode_hypotheses, returning programs in row order.

    Args:
        op_ids:    [K, L]    torch.long
        op_params: [K, L, P] torch.long

    Raises:
        TypeError: Inputs are not torch.long tensors.
        ValueError: Shapes disagree, IDs are unknown, operations do
            not form a prefix of each row, padded slots are nonzero,
            or decoded parameters are invalid for their operation.
    """

    op_mask, param_mask = operation_masks(op_ids)

    if op_ids.ndim != 2:
        raise ValueError("op_ids must have shape [K, L]")

    if not isinstance(op_params, Tensor):
        raise TypeError("op_params must be a PyTorch tensor")

    if op_params.dtype != torch.long:
        raise TypeError("op_params must have dtype torch.long")

    if tuple(op_params.shape) != (*op_ids.shape, PARAM_CAPACITY):
        raise ValueError(
            f"op_params must have shape "
            f"{(*op_ids.shape, PARAM_CAPACITY)}, "
            f"got {tuple(op_params.shape)}"
        )

    if (op_mask[:, 1:] & ~op_mask[:, :-1]).any().item():
        raise ValueError(
            "Each program's operations must precede its padding"
        )

    if (op_params[~param_mask] != PAD_PARAM_VALUE).any().item():
        raise ValueError(
            f"Unused parameter slots must equal {PAD_PARAM_VALUE}"
        )

    hypotheses = []

    for id_row, param_row in zip(op_ids.tolist(), op_params.tolist()):
        operations = []

        for op_id, params in zip(id_row, param_row):
            if op_id == PAD_OP_ID:
                break

            kind = OP_KINDS[op_id - 1]
            values = tuple(params[:OP_ARITY[kind]])

            if kind == "rotate":
                values = (values[0] * ROTATION_UNIT_DEGREES,)

            # Operation re-validates the decoded parameters.
            operations.append(Operation(kind, values))

        hypotheses.append(Hypothesis(tuple(operations)))

    return tuple(hypotheses)


def encoding_metadata() -> dict:
    """Describe the encoding as JSON-compatible values for records."""
    return {
        "version": ENCODING_VERSION,
        "op_kinds": list(OP_KINDS),
        "op_arity": dict(OP_ARITY),
        "pad_op_id": PAD_OP_ID,
        "param_capacity": PARAM_CAPACITY,
        "pad_param_value": PAD_PARAM_VALUE,
        "num_param_values": NUM_PARAM_VALUES,
        "rotation_unit_degrees": ROTATION_UNIT_DEGREES,
    }
