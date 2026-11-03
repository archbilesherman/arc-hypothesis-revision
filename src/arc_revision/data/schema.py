
"""Episode structure and tensor consistency checks."""

from dataclasses import dataclass

import torch
from torch import Tensor

from arc_revision.data.hypotheses import (
    NUM_COLORS,
    Hypothesis,
    validate_hypothesis_space,
)


GRID_HEIGHT = 5
GRID_WIDTH = 5
MIN_DEMOS = 2
MAX_DEMOS = 5


@dataclass
class ARCEpisode:
    """
    One ARC episode with T demonstrations and K candidate programs.

    Unbatched shapes:
        demonstrations: [T, 2, H, W] torch.long
        query_input:    [H, W]       torch.long
        query_output:   [H, W]       torch.long
        valid_masks:    [T, K]       torch.bool
        soft_targets:   [T, K]       torch.float32

    For now: 2 <= T <= 5, H = W = 5.
    K = len(hypotheses).

    rule_index identifies the ground-truth program within hypotheses.
    """

    demonstrations: Tensor
    query_input: Tensor
    query_output: Tensor
    rule_index: int
    valid_masks: Tensor
    soft_targets: Tensor
    hypotheses: tuple[Hypothesis, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate episode structure and hypothesis targets."""

        validate_hypothesis_space(self.hypotheses)
        num_rules = len(self.hypotheses)

        if not isinstance(self.demonstrations, Tensor):
            raise TypeError(
                "demonstrations must be a PyTorch tensor"
            )

        if self.demonstrations.ndim != 4:
            raise ValueError(
                "demonstrations must have four dimensions"
            )

        num_demos = self.demonstrations.shape[0]

        if not MIN_DEMOS <= num_demos <= MAX_DEMOS:
            raise ValueError(
                f"Episode must contain {MIN_DEMOS}-{MAX_DEMOS} demonstrations"
            )

        self._validate_grid(
            "demonstrations",
            self.demonstrations,
            (num_demos, 2, GRID_HEIGHT, GRID_WIDTH),
        )

        self._validate_grid(
            "query_input",
            self.query_input,
            (GRID_HEIGHT, GRID_WIDTH),
        )

        self._validate_grid(
            "query_output",
            self.query_output,
            (GRID_HEIGHT, GRID_WIDTH),
        )

        if type(self.rule_index) is not int:
            raise TypeError("rule_index must be an integer")

        if not 0 <= self.rule_index < num_rules:
            raise ValueError(
                f"rule_index must be between 0 and {num_rules - 1}"
            )

        self._validate_tensor(
            "valid_masks",
            self.valid_masks,
            (num_demos, num_rules),
            torch.bool,
        )

        self._validate_tensor(
            "soft_targets",
            self.soft_targets,
            (num_demos, num_rules),
            torch.float32,
        )

        if not torch.isfinite(self.soft_targets).all().item():
            raise ValueError(
                "soft_targets must contain finite values"
            )

        if (self.soft_targets < 0).any().item():
            raise ValueError(
                "soft_targets cannot be negative"
            )

        if not torch.allclose(
            self.soft_targets.sum(dim=-1),
            torch.ones(
                num_demos,
                device=self.soft_targets.device,
            ),
            atol=1e-5,
        ):
            raise ValueError(
                "soft_targets must sum to 1 for each demonstration"
            )

        if (self.soft_targets[~self.valid_masks] != 0).any().item():
            raise ValueError(
                "soft_targets must assign zero probability "
                "to invalid hypotheses"
            )

        if not self.valid_masks[:, self.rule_index].all().item():
            raise ValueError(
                "The ground-truth rule must remain valid"
            )

        # Version spaces may shrink but cannot expand.
        if (
            self.valid_masks[1:] & ~self.valid_masks[:-1]
        ).any().item():
            raise ValueError(
                "Previously eliminated hypotheses cannot reappear"
            )

    @staticmethod
    def _validate_tensor(
        name: str,
        value: Tensor,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
    ) -> None:
        """Validate tensor type, shape, and dtype."""

        if not isinstance(value, Tensor):
            raise TypeError(
                f"{name} must be a PyTorch tensor"
            )

        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )

        if value.dtype != expected_dtype:
            raise TypeError(
                f"{name} must have dtype {expected_dtype}, "
                f"got {value.dtype}"
            )

    @classmethod
    def _validate_grid(
        cls,
        name: str,
        value: Tensor,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Validate a grid tensor and its color values."""

        cls._validate_tensor(
            name,
            value,
            expected_shape,
            torch.long,
        )

        if ((value < 0) | (value >= NUM_COLORS)).any().item():
            raise ValueError(
                f"{name} must contain colors from "
                f"0 to {NUM_COLORS - 1}"
            )
