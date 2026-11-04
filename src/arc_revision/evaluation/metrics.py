"""Evaluation metrics for candidate-conditioned ARC predictions."""

import torch
from torch import Tensor


def final_candidate_indices(
    logits: Tensor,
    demo_mask: Tensor,
    candidate_mask: Tensor,
) -> Tensor:
    """Select the best real candidate at each episode's last real step."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, K]")

    batch_size, steps, candidates = logits.shape

    if demo_mask.shape != (batch_size, steps):
        raise ValueError("Invalid demo_mask shape")
    if candidate_mask.shape != (batch_size, candidates):
        raise ValueError("Invalid candidate_mask shape")
    if demo_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
        raise TypeError("Masks must be boolean")
    if not demo_mask.any(dim=1).all():
        raise ValueError("Every episode needs demonstrations")
    if not candidate_mask.any(dim=1).all():
        raise ValueError("Every episode needs candidates")
    if (demo_mask[:, 1:] & ~demo_mask[:, :-1]).any():
        raise ValueError("Demonstration masks must be prefixes")

    final_steps = demo_mask.sum(dim=1) - 1

    final_logits = logits[
        torch.arange(batch_size, device=logits.device),
        final_steps,
    ]

    final_logits = final_logits.masked_fill(
        ~candidate_mask, float("-inf")
    )

    if not torch.isfinite(final_logits[candidate_mask]).all():
        raise ValueError("Valid final candidate logits must be finite")

    return final_logits.argmax(dim=-1)


def program_correct(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    """Per-episode program classification correctness, [B] bool."""
    if predicted.ndim != 1 or predicted.shape != target.shape:
        raise ValueError("Program indices must both have shape [B]")

    return predicted == target


def grid_exact_matches(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    """Per-episode whole-grid exact match, [B] bool."""
    if predicted.ndim != 3 or predicted.shape != target.shape:
        raise ValueError("Grids must both have shape [B, H, W]")

    return (predicted == target).all(dim=(1, 2))


def valid_space_mass(
    logits: Tensor,
    valid_masks: Tensor,
    demo_mask: Tensor,
    candidate_mask: Tensor,
) -> Tensor:
    """Probability mass on valid hypotheses after each demonstration.

    Returns [B, T], with zero for padded demonstration positions.
    """
    if logits.ndim != 3 or valid_masks.shape != logits.shape:
        raise ValueError("Expected matching [B, T, K] tensors")

    batch_size, steps, candidates = logits.shape

    if demo_mask.shape != (batch_size, steps):
        raise ValueError("Invalid demo_mask shape")
    if candidate_mask.shape != (batch_size, candidates):
        raise ValueError("Invalid candidate_mask shape")
    if (
        valid_masks.dtype != torch.bool
        or demo_mask.dtype != torch.bool
        or candidate_mask.dtype != torch.bool
    ):
        raise TypeError("All masks must be boolean")

    if not candidate_mask.any(dim=1).all():
        raise ValueError("Each episode needs real candidates")

    # Inactive time steps and padded candidates are ignored.
    scores = logits.masked_fill(
        ~demo_mask.unsqueeze(-1), 0.0
    )
    scores = scores.masked_fill(
        ~candidate_mask.unsqueeze(1), float("-inf")
    )

    probabilities = torch.softmax(scores, dim=-1)

    valid = (
        valid_masks
        & demo_mask.unsqueeze(-1)
        & candidate_mask.unsqueeze(1)
    )

    mass = (
        probabilities * valid.to(probabilities.dtype)
    ).sum(dim=-1)

    return mass.masked_fill(~demo_mask, 0.0)