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


class CandidateBeliefHead(nn.Module):
    """Maintain a persistent recurrent state for each candidate program."""

    def __init__(
        self,
        demo_dim: int,
        task_dim: int,
        program_dim: int,
        belief_dim: int = 128,
    ) -> None:
        super().__init__()

        self.initial_state = nn.Linear(program_dim, belief_dim)
        self.cell = nn.GRUCell(
            demo_dim + task_dim + program_dim,
            belief_dim,
        )
        self.classifier = nn.Linear(belief_dim, 1)

    def forward(
        self,
        demo_embeddings: Tensor,
        task_states: Tensor,
        program_embeddings: Tensor,
        demo_mask: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        """Return candidate logits with shape [B, T, K]."""
        batch_size, num_steps, _ = demo_embeddings.shape
        num_candidates = program_embeddings.size(1)

        # Each program starts with its own state and shares the GRUCell weights.
        belief = torch.tanh(self.initial_state(program_embeddings))
        scores = []

        for step in range(num_steps):
            context = torch.cat(
                (demo_embeddings[:, step], task_states[:, step]),
                dim=-1,
            )
            context = context.unsqueeze(1).expand(
                -1, num_candidates, -1
            )
            cell_input = torch.cat(
                (context, program_embeddings), dim=-1
            )

            updated = self.cell(
                cell_input.reshape(batch_size * num_candidates, -1),
                belief.reshape(batch_size * num_candidates, -1),
            )
            updated = updated.reshape(batch_size, num_candidates, -1)

            # Padded demonstrations must not change candidate states.
            belief = torch.where(
                demo_mask[:, step, None, None],
                updated,
                belief,
            )

            logits = self.classifier(belief).squeeze(-1)
            logits = logits.masked_fill(~candidate_mask, float("-inf"))
            scores.append(logits)

        return torch.stack(scores, dim=1)


class SequentialGRUHypothesisClassifier(nn.Module):
    """Score candidate programs after each ARC demonstration."""

    def __init__(
        self,
        demonstration_dim: int = 128,
        hidden_dim: int = 128,
        program_dim: int = 128,
        num_gru_layers: int = 1,
        scoring_head: str = "dot_product",
        belief_dim: int = 128,
    ) -> None:
        super().__init__()
        if type(num_gru_layers) is not int or num_gru_layers < 1:
            raise ValueError("num_gru_layers must be a positive integer")
        if scoring_head not in ("dot_product", "candidate_gru"):
            raise ValueError(
                "scoring_head must be 'dot_product' or 'candidate_gru'"
            )

        self.demonstration_encoder = SharedCNNGridEncoder(
            embedding_dim=demonstration_dim
        )
        self.recurrent = nn.GRU(
            input_size=demonstration_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
            num_layers=num_gru_layers,
        )
        self.program_encoder = ProgramEncoder(
            embedding_dim=program_dim
        )

        self.scoring_head = scoring_head
        if scoring_head == "dot_product":
            # Original scoring path; retains its checkpoint parameter names.
            self.task_projection = nn.Linear(hidden_dim, program_dim)
            self.program_dim = program_dim
        else:
            self.belief_head = CandidateBeliefHead(
                demo_dim=demonstration_dim,
                task_dim=hidden_dim,
                program_dim=program_dim,
                belief_dim=belief_dim,
            )

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
            candidate_mask.ndim != 2
            or candidate_mask.shape != (batch, op_ids.shape[1])
            or candidate_mask.dtype != torch.bool
            or not candidate_mask.any(dim=1).all()
        ):
            raise ValueError(
                "candidate_mask must be boolean [B, K] "
                "with at least one candidate per episode"
            )

        # A06: encode demonstrations, including masked padded examples.
        demo_vectors = self.demonstration_encoder(
            demonstrations, demo_mask
        )

        # The original task GRU processes only real demonstration steps.
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

        # Each episode supplies its own candidate programs.
        program_vectors = self.program_encoder(
            op_ids,
            op_params,
            op_mask,
            param_mask,
            candidate_mask,
        )

        if self.scoring_head == "dot_product":
            task_vectors = self.task_projection(states)
            logits = torch.bmm(
                task_vectors,
                program_vectors.transpose(1, 2),
            ) / math.sqrt(self.program_dim)
            logits = logits.masked_fill(
                ~candidate_mask[:, None, :], float("-inf")
            )
        else:
            logits = self.belief_head(
                demo_embeddings=demo_vectors,
                task_states=states,
                program_embeddings=program_vectors,
                demo_mask=demo_mask,
                candidate_mask=candidate_mask,
            )

        # Padded time steps are not predictions. The loss must ignore them.
        return logits.masked_fill(~demo_mask[:, :, None], 0.0)
