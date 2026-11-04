"""Order-sensitive encoding of ARC candidate programs."""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from arc_revision.data.encoding import (
    NUM_OP_IDS,
    NUM_PARAM_VALUES,
    PAD_OP_ID,
    PARAM_CAPACITY,
)


class ProgramEncoder(nn.Module):
    """Encode [B, K, L] operation sequences into [B, K, D]."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding_dim = embedding_dim

        self.operation_embedding = nn.Embedding(
            NUM_OP_IDS,
            embedding_dim,
            padding_idx=PAD_OP_ID,
        )

        # Each (parameter slot, value) gets a distinct embedding.
        self.parameter_embedding = nn.Embedding(
            PARAM_CAPACITY * NUM_PARAM_VALUES,
            embedding_dim,
        )

        self.operation_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
        )

        # Learned representation of the empty identity program.
        self.identity_embedding = nn.Parameter(
            torch.randn(embedding_dim) * 0.02
        )

    def forward(
        self,
        op_ids: Tensor,
        op_params: Tensor,
        op_mask: Tensor,
        param_mask: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        if op_ids.ndim != 3:
            raise ValueError("op_ids must have shape [B, K, L]")

        batch, candidates, length = op_ids.shape

        if length < 1:
            raise ValueError("Program length dimension must be >= 1")

        if op_params.shape != (
            batch, candidates, length, PARAM_CAPACITY
        ):
            raise ValueError("Invalid op_params shape")

        if op_mask.shape != op_ids.shape:
            raise ValueError("Invalid op_mask shape")

        if param_mask.shape != op_params.shape:
            raise ValueError("Invalid param_mask shape")

        if candidate_mask.shape != (batch, candidates):
            raise ValueError("Invalid candidate_mask shape")

        if op_ids.dtype != torch.long or op_params.dtype != torch.long:
            raise TypeError("Operation IDs and parameters must be long")

        if any(
            mask.dtype != torch.bool
            for mask in (op_mask, param_mask, candidate_mask)
        ):
            raise TypeError("Masks must be boolean")

        if (op_mask[..., 1:] & ~op_mask[..., :-1]).any():
            raise ValueError("Operation masks must be prefixes")

        if (param_mask & ~op_mask.unsqueeze(-1)).any():
            raise ValueError("Padded operations cannot have parameters")

        if ((op_ids < 0) | (op_ids >= NUM_OP_IDS)).any():
            raise ValueError("Operation ID out of range")

        valid_params = op_params[param_mask]

        if (
            (valid_params < 0) |
            (valid_params >= NUM_PARAM_VALUES)
        ).any():
            raise ValueError("Parameter value out of range")

        # Ignore values stored in unused parameter slots.
        safe_params = op_params.masked_fill(~param_mask, 0)

        positions = torch.arange(
            PARAM_CAPACITY, device=op_ids.device
        )

        # Position-aware parameter token IDs.
        parameter_tokens = (
            safe_params + positions * NUM_PARAM_VALUES
        )

        parameter_vectors = self.parameter_embedding(
            parameter_tokens
        )

        parameter_vectors = (
            parameter_vectors * param_mask.unsqueeze(-1)
        ).sum(dim=-2)

        operation_vectors = (
            self.operation_embedding(op_ids) +
            parameter_vectors
        )

        operation_vectors = (
            operation_vectors * op_mask.unsqueeze(-1)
        )

        # Treat each candidate as an independent operation sequence.
        flat = operation_vectors.reshape(
            batch * candidates, length, self.embedding_dim
        )

        lengths = op_mask.sum(dim=-1).reshape(-1)

        # Packed sequences cannot have length zero.
        packed = pack_padded_sequence(
            flat,
            lengths.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.operation_gru(packed)

        embeddings = hidden[-1].reshape(
            batch, candidates, self.embedding_dim
        )

        # Empty programs use a learned identity representation.
        embeddings = torch.where(
            lengths.reshape(batch, candidates, 1) == 0,
            self.identity_embedding.view(1, 1, -1),
            embeddings,
        )

        return embeddings.masked_fill(
            ~candidate_mask.unsqueeze(-1), 0
        )