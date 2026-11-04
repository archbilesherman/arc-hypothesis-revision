"""Masked final-only and revision-aware objectives for candidate programs."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_layout(
    logits: Tensor,
    demo_mask: Tensor,
    candidate_mask: Tensor,
) -> None:
    """Validate shapes, masking, and real candidate scores."""
    if logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("logits must be floating-point [B, T, K]")

    batch_size, steps, candidates = logits.shape

    if batch_size == 0 or steps == 0 or candidates == 0:
        raise ValueError(
            "batch, demonstration, and candidate dimensions must be nonempty"
        )

    if demo_mask.shape != (batch_size, steps) or demo_mask.dtype != torch.bool:
        raise ValueError("demo_mask must be boolean [B, T]")

    if (
        candidate_mask.shape != (batch_size, candidates)
        or candidate_mask.dtype != torch.bool
    ):
        raise ValueError("candidate_mask must be boolean [B, K]")

    if (
        logits.device != demo_mask.device
        or logits.device != candidate_mask.device
    ):
        raise ValueError("logits and masks must be on the same device")

    if (
        not demo_mask[:, 0].all()
        or (demo_mask[:, 1:] & ~demo_mask[:, :-1]).any()
    ):
        raise ValueError(
            "demo_mask must be a nonempty prefix per episode"
        )

    if not candidate_mask.any(dim=1).all():
        raise ValueError("Each episode needs at least one candidate")

    valid = demo_mask.unsqueeze(-1) & candidate_mask.unsqueeze(1)

    if not torch.isfinite(logits[valid]).all():
        raise ValueError("Valid logits must be finite")


def final_only_loss(
    logits: Tensor,
    rule_index: Tensor,
    demo_mask: Tensor,
    candidate_mask: Tensor,
) -> Tensor:
    """Mean ground-truth CE at each episode's final real demonstration."""
    _validate_layout(logits, demo_mask, candidate_mask)

    batch_size, _, candidates = logits.shape

    if (
        rule_index.shape != (batch_size,)
        or rule_index.dtype != torch.long
    ):
        raise ValueError("rule_index must be torch.long [B]")

    if rule_index.device != logits.device:
        raise ValueError("rule_index must be on the logits device")

    if ((rule_index < 0) | (rule_index >= candidates)).any():
        raise ValueError("rule_index out of range")

    if not candidate_mask.gather(1, rule_index.unsqueeze(1)).all():
        raise ValueError("rule_index points to a padded candidate")

    # Each episode may have a different final demonstration.
    final_steps = demo_mask.sum(dim=1) - 1

    final_logits = logits[
        torch.arange(batch_size, device=logits.device),
        final_steps,
    ]

    # Padded candidates must not enter probability normalization.
    final_logits = final_logits.masked_fill(
        ~candidate_mask,
        float("-inf"),
    )

    return F.cross_entropy(final_logits, rule_index)


def revision_loss(
    logits: Tensor,
    soft_targets: Tensor,
    demo_mask: Tensor,
    candidate_mask: Tensor,
) -> Tensor:
    """Average soft-target CE over steps per episode, then episodes.

    The final valid demonstration is included. Padded candidates and
    demonstrations contribute neither probability mass nor gradients.
    """
    _validate_layout(logits, demo_mask, candidate_mask)

    if (
        soft_targets.shape != logits.shape
        or not soft_targets.is_floating_point()
    ):
        raise ValueError(
            "soft_targets must be floating-point [B, T, K]"
        )

    if soft_targets.device != logits.device:
        raise ValueError("soft_targets must be on the logits device")

    valid = demo_mask.unsqueeze(-1) & candidate_mask.unsqueeze(1)

    # Ignore padded targets, including arbitrary sentinel values.
    targets = soft_targets.masked_fill(~valid, 0.0)

    if (
        not torch.isfinite(targets).all()
        or (targets < 0).any()
    ):
        raise ValueError(
            "Valid soft_targets must be finite and nonnegative"
        )

    masses = targets.sum(dim=-1)[demo_mask]

    if not torch.allclose(
        masses,
        torch.ones_like(masses),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            "soft_targets must sum to one at each valid step"
        )

    # Neutralize padded demonstrations before log_softmax.
    safe_logits = logits.masked_fill(
        ~demo_mask.unsqueeze(-1), 0.0
    )

    safe_logits = safe_logits.masked_fill(
        ~candidate_mask.unsqueeze(1),
        float("-inf"),
    )

    log_probs = F.log_softmax(safe_logits, dim=-1)

    # Avoid 0 * -inf = NaN for padded candidates.
    log_probs = log_probs.masked_fill(
        ~candidate_mask.unsqueeze(1), 0.0
    )

    per_step = -(targets * log_probs).sum(dim=-1)

    # Equal weighting across episodes, regardless of sequence length.
    per_episode = (
        per_step.sum(dim=1) / demo_mask.sum(dim=1)
    )

    return per_episode.mean()


def revision_aware_loss(
    logits: Tensor,
    rule_index: Tensor,
    demo_mask: Tensor,
    candidate_mask: Tensor,
    soft_targets: Tensor,
    *,
    lambda_revision: float = 1.0,
) -> Tensor:
    """Final-only loss plus lambda_revision times revision CE."""
    if (
        isinstance(lambda_revision, bool)
        or not isinstance(lambda_revision, (int, float))
    ):
        raise TypeError(
            "lambda_revision must be a finite nonnegative number"
        )

    if (
        not math.isfinite(lambda_revision)
        or lambda_revision < 0
    ):
        raise ValueError(
            "lambda_revision must be finite and nonnegative"
        )

    final = final_only_loss(
        logits, rule_index, demo_mask, candidate_mask
    )

    if lambda_revision == 0:
        return final

    auxiliary = revision_loss(
        logits, soft_targets, demo_mask, candidate_mask
    )

    return final + lambda_revision * auxiliary