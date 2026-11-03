"""Shared CNN encoder for ARC input-output demonstration sequences."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SharedCNNGridEncoder(nn.Module):
    """Map integer ARC demonstration pairs to fixed-size embeddings.

    Input:  demonstrations [B, T, 2, H, W] with colors in [0, 9]
    Mask:   demonstration_mask [B, T] (True for valid demonstrations)
    Output: embeddings [B, T, D], zeroed at padded positions
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        grid_embedding_dim: int = 64,
        hidden_channels: int = 32,
        grid_size: int = 5,
    ) -> None:
        super().__init__()
        if min(embedding_dim, grid_embedding_dim, hidden_channels, grid_size) <= 0:
            raise ValueError("All encoder dimensions must be positive")

        self.grid_size = grid_size
        self.grid_embedding_dim = grid_embedding_dim

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, 2 * hidden_channels,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                2 * hidden_channels * grid_size * grid_size,
                grid_embedding_dim,
            ),
            nn.ReLU(),
        )

        self.pair_projection = nn.Linear(
            2 * grid_embedding_dim, embedding_dim
        )

    def forward(
        self,
        demonstrations: Tensor,
        demonstration_mask: Tensor | None = None,
    ) -> Tensor:
        if demonstrations.ndim != 5 or demonstrations.shape[2] != 2:
            raise ValueError(
                "Expected demonstrations with shape [B, T, 2, H, W]"
            )

        batch_size, steps, _, height, width = demonstrations.shape

        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError(
                f"Expected {self.grid_size}x{self.grid_size} grids"
            )

        if demonstrations.dtype not in (
            torch.uint8, torch.int8, torch.int16,
            torch.int32, torch.int64,
        ):
            raise TypeError("ARC grids must contain integer color IDs")

        if demonstration_mask is not None:
            if demonstration_mask.shape != (batch_size, steps):
                raise ValueError(
                    "Expected demonstration_mask with shape [B, T]"
                )

            if demonstration_mask.dtype != torch.bool:
                raise TypeError(
                    "demonstration_mask must be a boolean tensor"
                )

            # Mask before one-hot encoding so padded values are safe.
            demonstrations = demonstrations.masked_fill(
                ~demonstration_mask[:, :, None, None, None], 0
            )

        if torch.any((demonstrations < 0) | (demonstrations > 9)):
            raise ValueError(
                "Unmasked ARC colors must be in the range 0..9"
            )

        # Flatten B, T, and the input/output pair dimension.
        grids = demonstrations.reshape(
            batch_size * steps * 2, height, width
        )

        # Integer colors -> one-hot CNN channels.
        colors = F.one_hot(grids.long(), num_classes=10)
        colors = colors.permute(0, 3, 1, 2).to(
            dtype=self.grid_encoder[0].weight.dtype
        )

        # Both grids pass through the same learned CNN.
        grid_embeddings = self.grid_encoder(colors)

        # Concatenate input and output representations per example.
        pairs = grid_embeddings.reshape(
            batch_size, steps, 2 * self.grid_embedding_dim
        )

        # Produce one embedding for each demonstration pair.
        embeddings = self.pair_projection(pairs)

        # Padded demonstrations must have zero embeddings.
        if demonstration_mask is not None:
            embeddings = embeddings.masked_fill(
                ~demonstration_mask.unsqueeze(-1), 0
            )

        return embeddings
