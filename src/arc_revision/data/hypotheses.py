
"""Executable ARC programs built from validated, parameterized operations."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from arc_revision.data.transforms import apply_transform


NUM_COLORS = 10

OpKind = Literal[
    "rotate",
    "reflect_horizontal",
    "reflect_vertical",
    "reflect_main_diagonal",
    "reflect_anti_diagonal",
    "replace_color",
    "swap_colors",
    "permute_colors",
]

# Internal mapping to the existing geometric primitive implementations.
GEOMETRY_RULES = {
    "reflect_horizontal": 4,
    "reflect_vertical": 5,
    "reflect_main_diagonal": 6,
    "reflect_anti_diagonal": 7,
}


@dataclass(frozen=True)
class Operation:
    """One DSL instruction with immutable positional parameters."""

    kind: OpKind
    params: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.params, tuple) or any(
            type(value) is not int for value in self.params
        ):
            raise TypeError("params must be a tuple of integers")

        if self.kind == "rotate":
            if len(self.params) != 1 or self.params[0] not in (90, 180, 270):
                raise ValueError("rotate requires 90, 180, or 270 degrees")

        elif self.kind in GEOMETRY_RULES:
            if self.params:
                raise ValueError(f"{self.kind} takes no parameters")

        elif self.kind in ("replace_color", "swap_colors"):
            if len(self.params) != 2 or any(
                not 0 <= value < NUM_COLORS for value in self.params
            ):
                raise ValueError(
                    f"{self.kind} requires two colors from 0 to 9"
                )

            if self.params[0] == self.params[1]:
                raise ValueError("source and target colors must differ")

            if self.kind == "swap_colors" and self.params[0] > self.params[1]:
                raise ValueError(
                    "swap_colors parameters must be ascending"
                )

        elif self.kind == "permute_colors":
            if len(self.params) != NUM_COLORS or set(self.params) != set(
                range(NUM_COLORS)
            ):
                raise ValueError(
                    "permute_colors requires a permutation of 0-9"
                )

        else:
            raise ValueError(f"Unknown operation: {self.kind}")


@dataclass(frozen=True)
class Hypothesis:
    """An ordered executable program; an empty program is identity."""

    operations: tuple[Operation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operations, tuple) or any(
            not isinstance(op, Operation) for op in self.operations
        ):
            raise TypeError(
                "operations must be a tuple of Operation objects"
            )


def validate_hypothesis_space(
    hypotheses: tuple[Hypothesis, ...],
) -> None:
    """Validate an ordered, nonempty candidate program set."""

    if not isinstance(hypotheses, tuple) or not hypotheses:
        raise ValueError("hypotheses must be a nonempty tuple")

    if any(not isinstance(h, Hypothesis) for h in hypotheses):
        raise TypeError(
            "hypotheses must contain Hypothesis objects"
        )

    if len(set(hypotheses)) != len(hypotheses):
        raise ValueError(
            "hypotheses must not contain duplicate programs"
        )


def apply_hypothesis(
    grid: Tensor,
    hypothesis: Hypothesis,
) -> Tensor:
    """Execute operations in order without modifying the input grid."""

    if not isinstance(hypothesis, Hypothesis):
        raise TypeError("hypothesis must be a Hypothesis")

    # Validates the input grid and creates an independent copy.
    result = apply_transform(grid, 0)

    for op in hypothesis.operations:
        if op.kind == "rotate":
            result = apply_transform(
                result, op.params[0] // 90
            )

        elif op.kind in GEOMETRY_RULES:
            result = apply_transform(
                result, GEOMETRY_RULES[op.kind]
            )

        elif op.kind == "replace_color":
            source, target = op.params
            result = torch.where(
                result == source, target, result
            )

        elif op.kind == "swap_colors":
            first, second = op.params
            result = torch.where(
                result == first,
                second,
                torch.where(result == second, first, result),
            )

        elif op.kind == "permute_colors":
            lookup = torch.tensor(
                op.params,
                device=result.device,
                dtype=torch.long,
            )
            result = lookup[result]

    return result