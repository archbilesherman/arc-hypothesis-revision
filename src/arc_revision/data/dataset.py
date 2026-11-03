"""PyTorch Dataset and DataLoader over saved synthetic ARC episodes.

Typical use:

    train = ARCEpisodeDataset.from_snapshot("data/generated/v1", "train")
    loader = make_dataloader(train, batch_size=64, shuffle=True, seed=0)

    for batch in loader:
        batch = batch.to(device)

Episodes keep their snapshot order, and candidates keep their saved
order, so rule_index always refers to the same program. collate_episodes
pads every variable-length dimension and returns explicit masks: the
padding values (colour 0, operation ID 0, parameter 0) are also real
values, so validity must always be read from the masks.
"""

import json
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from arc_revision.data.encoding import (
    PAD_OP_ID,
    PAD_PARAM_VALUE,
    PARAM_CAPACITY,
    decode_hypotheses,
    encode_hypotheses,
    operation_masks,
)
from arc_revision.data.generator import SPLIT_OFFSETS, load_split
from arc_revision.data.schema import (
    GRID_HEIGHT,
    GRID_WIDTH,
    ARCEpisode,
)


Split = Literal["train", "validation", "test"]
SPLITS: tuple[str, ...] = tuple(SPLIT_OFFSETS)

GRID_PAD_VALUE = 0

# Grids hold colours 0-9 and program encodings hold IDs and parameters
# below 10, so they are stored as uint8 and widened on access.
_STORAGE_DTYPE = torch.uint8


@dataclass(frozen=True)
class EpisodeSample:
    """
    One episode as unpadded, correctly typed tensors.

    Shapes:
        demonstrations: [T, 2, H, W] torch.long
        query_input:    [H, W]       torch.long
        query_output:   [H, W]       torch.long
        valid_masks:    [T, K]       torch.bool
        soft_targets:   [T, K]       torch.float32
        op_ids:         [K, L]       torch.long
        op_params:      [K, L, P]    torch.long

    rule_index indexes the ground-truth program among the K candidates;
    episode_index is the episode's position in its dataset.
    """

    demonstrations: Tensor
    query_input: Tensor
    query_output: Tensor
    rule_index: int
    valid_masks: Tensor
    soft_targets: Tensor
    op_ids: Tensor
    op_params: Tensor
    episode_index: int


@dataclass(frozen=True)
class EpisodeBatch:
    """
    A padded batch of B episodes.

    Shapes, with T = T_max, K = K_max, L = L_max, P = PARAM_CAPACITY:
        demonstrations: [B, T, 2, H, W] torch.long     pad 0
        demo_mask:      [B, T]          torch.bool
        query_input:    [B, H, W]       torch.long
        query_output:   [B, H, W]       torch.long
        rule_index:     [B]             torch.long
        valid_masks:    [B, T, K]       torch.bool     pad False
        soft_targets:   [B, T, K]       torch.float32  pad 0
        candidate_mask: [B, K]          torch.bool
        op_ids:         [B, K, L]       torch.long     pad PAD_OP_ID
        op_params:      [B, K, L, P]    torch.long     pad PAD_PARAM_VALUE
        op_mask:        [B, K, L]       torch.bool
        param_mask:     [B, K, L, P]    torch.bool
        episode_index:  [B]             torch.long

    Each real demonstration's soft_targets row sums to 1 over real
    candidates. Rows for padded demonstrations are all zero, so they
    must be excluded with demo_mask. The last real step of episode b
    is num_demos[b] - 1.
    """

    demonstrations: Tensor
    demo_mask: Tensor
    query_input: Tensor
    query_output: Tensor
    rule_index: Tensor
    valid_masks: Tensor
    soft_targets: Tensor
    candidate_mask: Tensor
    op_ids: Tensor
    op_params: Tensor
    op_mask: Tensor
    param_mask: Tensor
    episode_index: Tensor

    @property
    def batch_size(self) -> int:
        """Number of episodes B."""
        return self.rule_index.shape[0]

    @property
    def num_demos(self) -> Tensor:
        """[B] long: real demonstrations per episode."""
        return self.demo_mask.sum(dim=-1)

    @property
    def num_candidates(self) -> Tensor:
        """[B] long: real candidate programs per episode."""
        return self.candidate_mask.sum(dim=-1)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "EpisodeBatch":
        """Return a copy with every tensor moved to device."""
        return replace(self, **{
            field.name: getattr(self, field.name).to(
                device, non_blocking=non_blocking
            )
            for field in fields(self)
        })


class ARCEpisodeDataset(Dataset[EpisodeSample]):
    """
    Map-style dataset over a fixed sequence of ARC episodes.

    Episodes are encoded once at construction and kept in a few large
    flat tensors addressed by per-episode offsets. Few large tensors
    are cheap to share with DataLoader worker processes, whereas
    thousands of small tensors would each need their own shared-memory
    segment.
    """

    def __init__(
        self,
        episodes: Iterable[ARCEpisode],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        demonstrations = []
        query_inputs = []
        query_outputs = []
        rule_indices = []
        valid_masks = []
        soft_targets = []
        op_ids = []
        op_params = []
        layout = []

        demo_start = cell_start = slot_start = 0

        for episode in episodes:
            if not isinstance(episode, ARCEpisode):
                raise TypeError(
                    "episodes must contain ARCEpisode objects"
                )

            programs = encode_hypotheses(episode.hypotheses)
            num_demos, num_candidates = episode.valid_masks.shape
            length = programs.op_ids.shape[1]

            demonstrations.append(_compact(episode.demonstrations))
            query_inputs.append(_compact(episode.query_input[None]))
            query_outputs.append(_compact(episode.query_output[None]))
            rule_indices.append(episode.rule_index)
            valid_masks.append(episode.valid_masks.cpu().flatten())
            soft_targets.append(episode.soft_targets.cpu().flatten())
            op_ids.append(_compact(programs.op_ids.flatten()))
            op_params.append(_compact(programs.op_params.flatten(0, 1)))

            layout.append((
                demo_start,
                cell_start,
                slot_start,
                num_demos,
                num_candidates,
                length,
            ))

            demo_start += num_demos
            cell_start += num_demos * num_candidates
            slot_start += num_candidates * length

        grid = (GRID_HEIGHT, GRID_WIDTH)

        self._demonstrations = _concat(demonstrations, (0, 2, *grid))
        self._query_inputs = _concat(query_inputs, (0, *grid))
        self._query_outputs = _concat(query_outputs, (0, *grid))
        self._rule_indices = torch.tensor(rule_indices, dtype=torch.long)
        self._valid_masks = _concat(valid_masks, (0,), torch.bool)
        self._soft_targets = _concat(soft_targets, (0,), torch.float32)
        self._op_ids = _concat(op_ids, (0,))
        self._op_params = _concat(op_params, (0, PARAM_CAPACITY))
        self._layout = torch.tensor(layout, dtype=torch.long).reshape(-1, 6)

        self.metadata: dict[str, Any] = dict(metadata or {})

    @classmethod
    def from_snapshot(
        cls,
        directory: str | Path,
        split: Split,
    ) -> "ARCEpisodeDataset":
        """
        Load one split of a saved snapshot, verifying its checksum
        and episode count against the snapshot manifest.
        """

        if split not in SPLIT_OFFSETS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {SPLITS}"
            )

        directory = Path(directory)
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        info = manifest["splits"][split]

        dataset = cls(
            load_split(directory, split),
            metadata={
                "directory": str(directory),
                "split": split,
                "format_version": manifest["format_version"],
                "config": manifest["config"],
                "count": info["count"],
                "sha256": info["sha256"],
            },
        )

        if len(dataset) != info["count"]:
            raise ValueError(
                f"{split} manifest lists {info['count']} episodes, "
                f"but the snapshot contains {len(dataset)}"
            )

        return dataset

    def __len__(self) -> int:
        return self._layout.shape[0]

    def __getitem__(self, index: int) -> EpisodeSample:
        # Normalizes negative indices and raises IndexError when out
        # of range, so episode_index is always the canonical position.
        index = range(len(self))[operator.index(index)]

        (
            demo_start,
            cell_start,
            slot_start,
            num_demos,
            num_candidates,
            length,
        ) = self._layout[index].tolist()

        cells = slice(cell_start, cell_start + num_demos * num_candidates)
        slots = slice(slot_start, slot_start + num_candidates * length)

        return EpisodeSample(
            demonstrations=self._demonstrations[
                demo_start:demo_start + num_demos
            ].long(),
            query_input=self._query_inputs[index].long(),
            query_output=self._query_outputs[index].long(),
            rule_index=int(self._rule_indices[index]),
            valid_masks=self._valid_masks[cells]
            .reshape(num_demos, num_candidates)
            .clone(),
            soft_targets=self._soft_targets[cells]
            .reshape(num_demos, num_candidates)
            .clone(),
            op_ids=self._op_ids[slots]
            .reshape(num_candidates, length)
            .long(),
            op_params=self._op_params[slots]
            .reshape(num_candidates, length, PARAM_CAPACITY)
            .long(),
            episode_index=index,
        )

    def episode(self, index: int) -> ARCEpisode:
        """Reconstruct the validated ARCEpisode stored at index."""

        sample = self[index]

        return ARCEpisode(
            demonstrations=sample.demonstrations,
            query_input=sample.query_input,
            query_output=sample.query_output,
            rule_index=sample.rule_index,
            valid_masks=sample.valid_masks,
            soft_targets=sample.soft_targets,
            hypotheses=decode_hypotheses(
                sample.op_ids,
                sample.op_params,
            ),
        )


def load_datasets(
    directory: str | Path,
    splits: Sequence[Split] = SPLITS,
) -> dict[str, ARCEpisodeDataset]:
    """Load several splits of one saved snapshot, keyed by split."""
    return {
        split: ARCEpisodeDataset.from_snapshot(directory, split)
        for split in splits
    }


def collate_episodes(samples: Sequence[EpisodeSample]) -> EpisodeBatch:
    """
    Pad and stack episodes into an EpisodeBatch.

    Every variable dimension (T, K, L) is right-padded to the batch
    maximum. Real entries are copied unchanged, so slicing a batch
    back to an episode's own sizes recovers its sample exactly.
    """

    if len(samples) == 0:
        raise ValueError("Cannot collate an empty batch")

    for sample in samples:
        _check_sample(sample)

    demo_counts = torch.tensor(
        [s.demonstrations.shape[0] for s in samples],
        dtype=torch.long,
    )
    candidate_counts = torch.tensor(
        [s.op_ids.shape[0] for s in samples],
        dtype=torch.long,
    )

    op_ids = _pad_stack([s.op_ids for s in samples], PAD_OP_ID)
    op_mask, param_mask = operation_masks(op_ids)

    return EpisodeBatch(
        demonstrations=_pad_stack(
            [s.demonstrations for s in samples],
            GRID_PAD_VALUE,
        ),
        demo_mask=_length_mask(demo_counts),
        query_input=torch.stack([s.query_input for s in samples]),
        query_output=torch.stack([s.query_output for s in samples]),
        rule_index=torch.tensor(
            [s.rule_index for s in samples],
            dtype=torch.long,
        ),
        valid_masks=_pad_stack([s.valid_masks for s in samples], False),
        soft_targets=_pad_stack([s.soft_targets for s in samples], 0.0),
        candidate_mask=_length_mask(candidate_counts),
        op_ids=op_ids,
        op_params=_pad_stack(
            [s.op_params for s in samples],
            PAD_PARAM_VALUE,
        ),
        op_mask=op_mask,
        param_mask=param_mask,
        episode_index=torch.tensor(
            [s.episode_index for s in samples],
            dtype=torch.long,
        ),
    )


def make_dataloader(
    dataset: ARCEpisodeDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader whose batch order is determined by seed alone.

    Shuffling draws from a dedicated generator, so it ignores global
    RNG state and repeats exactly across runs, while successive epochs
    still see different orders. Samples are fixed tensors, so worker
    processes need no seeding of their own.
    """

    if not isinstance(dataset, ARCEpisodeDataset):
        raise TypeError("dataset must be an ARCEpisodeDataset")

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_episodes,
        generator=generator,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _compact(tensor: Tensor) -> Tensor:
    """Copy small non-negative integers to compact CPU storage."""
    return tensor.to("cpu", _STORAGE_DTYPE)


def _concat(
    chunks: list[Tensor],
    empty_shape: tuple[int, ...],
    dtype: torch.dtype = _STORAGE_DTYPE,
) -> Tensor:
    """Concatenate chunks along dim 0; an empty list gives empty_shape."""
    if not chunks:
        return torch.empty(empty_shape, dtype=dtype)
    return torch.cat(chunks)


def _check_sample(sample: EpisodeSample) -> None:
    """Reject samples whose per-episode sizes disagree."""

    if not isinstance(sample, EpisodeSample):
        raise TypeError("samples must be EpisodeSample objects")

    num_demos = sample.demonstrations.shape[0]
    num_candidates, length = sample.op_ids.shape

    expected = {
        "valid_masks": (num_demos, num_candidates),
        "soft_targets": (num_demos, num_candidates),
        "op_params": (num_candidates, length, PARAM_CAPACITY),
    }

    for name, shape in expected.items():
        actual = tuple(getattr(sample, name).shape)

        if actual != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {actual}"
            )


def _length_mask(lengths: Tensor) -> Tensor:
    """[B] lengths -> [B, max_length] bool mask of real positions."""
    positions = torch.arange(int(lengths.max()))
    return positions < lengths[:, None]


def _pad_stack(
    tensors: Sequence[Tensor],
    value: bool | int | float,
) -> Tensor:
    """Stack equal-rank tensors, right-padding each dimension."""

    shape = tuple(
        max(sizes) for sizes in zip(*(t.shape for t in tensors))
    )
    out = tensors[0].new_full((len(tensors), *shape), value)

    for row, tensor in zip(out, tensors):
        row[tuple(slice(0, size) for size in tensor.shape)] = tensor

    return out
