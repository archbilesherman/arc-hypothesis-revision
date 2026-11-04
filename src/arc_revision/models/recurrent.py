"""Sequential GRU hypothesis classifier."""

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from arc_revision.models.encoder import SharedCNNGridEncoder
from arc_revision.models.program_encoder import ProgramEncoder


class SequentialGRUHypothesisClassifier(nn.Module):
    """Score candidate programs after each ARC demonstration."""

    def __init__(
        self,
        demonstration_dim: int = 128,
        hidden_dim: int = 128,
        program_dim: int = 128,
    ) -> None:
        super().__init__()

        self.demonstration_encoder = SharedCNNGridEncoder(
            embedding_dim=demonstration_dim
        )

        self.recurrent = nn.GRU(
            input_size=demonstration_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.program_encoder = ProgramEncoder(
            embedding_dim=program_dim
        )

        self.task_projection = nn.Linear(
            hidden_dim, program_dim
        )

        self.program_dim = program_dim

    def forward(
        self,
        demonstrations: Tensor,
        demo_mask: Tensor,
        op_ids: Tensor,
        op_params: Tensor,
        op_mask: Tensor,
        param_mask: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        if demonstrations.ndim != 5:
            raise ValueError(
                "demonstrations must have shape [B, T, 2, H, W]"
            )

        batch, steps = demonstrations.shape[:2]

        if (
            demo_mask.shape != (batch, steps)
            or demo_mask.dtype != torch.bool
        ):
            raise ValueError("demo_mask must be boolean [B, T]")

        if not demo_mask.any(dim=1).all():
            raise ValueError(
                "Each episode needs at least one demonstration"
            )

        if (demo_mask[:, 1:] & ~demo_mask[:, :-1]).any():
            raise ValueError("demo_mask must be a prefix mask")

        if (
            candidate_mask.shape[0] != batch
            or not candidate_mask.any(dim=1).all()
        ):
            raise ValueError(
                "Each episode needs at least one candidate"
            )

        # Encode each demonstration using A06.
        demo_vectors = self.demonstration_encoder(
            demonstrations, demo_mask
        )

        # Process only real demonstration time steps.
        packed = pack_padded_sequence(
            demo_vectors,
            demo_mask.sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_states, _ = self.recurrent(packed)

        states, _ = pad_packed_sequence(
            packed_states,
            batch_first=True,
            total_length=steps,
        )

        # Encode the episode-specific candidate programs.
        program_vectors = self.program_encoder(
            op_ids,
            op_params,
            op_mask,
            param_mask,
            candidate_mask,
        )

        task_vectors = self.task_projection(states)

        # Score each program against each recurrent task state.
        logits = torch.bmm(
            task_vectors,
            program_vectors.transpose(1, 2),
        ) / math.sqrt(self.program_dim)

        # Invalid candidates cannot be selected.
        logits = logits.masked_fill(
            ~candidate_mask[:, None, :],
            float("-inf"),
        )

        # Padded time steps are not predictions.
        # A08 must also exclude them from the loss.
        return logits.masked_fill(
            ~demo_mask[:, :, None], 0.0
        )