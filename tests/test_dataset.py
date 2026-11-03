"""Tests for the ARC episode Dataset, collation, and DataLoader."""

import json
from dataclasses import fields, replace

import pytest
import torch

from arc_revision.data.dataset import (
    SPLITS,
    ARCEpisodeDataset,
    EpisodeBatch,
    EpisodeSample,
    collate_episodes,
    load_datasets,
    make_dataloader,
)
from arc_revision.data.encoding import (
    NUM_OP_IDS,
    NUM_PARAM_VALUES,
    PAD_OP_ID,
    PAD_PARAM_VALUE,
    PARAM_CAPACITY,
    decode_hypotheses,
    operation_masks,
)
from arc_revision.data.generator import (
    GenerationConfig,
    load_split,
    save_dataset,
)
from arc_revision.data.hypotheses import NUM_COLORS, Hypothesis
from arc_revision.data.schema import ARCEpisode


# 24 training episodes give six of each demonstration count (2-5).
CONFIG = GenerationConfig(train_size=24, validation_size=8, test_size=8)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    directory = tmp_path_factory.mktemp("snapshot") / "v1"
    save_dataset(directory, CONFIG)
    return directory


@pytest.fixture(scope="module")
def train(snapshot):
    return ARCEpisodeDataset.from_snapshot(snapshot, "train")


@pytest.fixture(scope="module")
def full_batch(train):
    return collate_episodes([train[i] for i in range(len(train))])


def assert_batches_equal(first: EpisodeBatch, second: EpisodeBatch):
    for field in fields(EpisodeBatch):
        a = getattr(first, field.name)
        b = getattr(second, field.name)
        assert a.dtype == b.dtype, field.name
        assert torch.equal(a, b), field.name


def test_loads_all_splits(snapshot):
    datasets = load_datasets(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())

    assert tuple(datasets) == SPLITS == ("train", "validation", "test")

    for split, dataset in datasets.items():
        info = manifest["splits"][split]

        assert len(dataset) == info["count"]
        assert dataset.metadata["split"] == split
        assert dataset.metadata["sha256"] == info["sha256"]
        assert dataset.metadata["config"] == manifest["config"]


def test_preserves_episode_data_and_order(snapshot, train):
    originals = list(load_split(snapshot, "train"))

    assert len(train) == len(originals)

    for index, original in enumerate(originals):
        restored = train.episode(index)

        # Candidate order is unchanged, so rule_index still points at
        # the same ground-truth program.
        assert restored.hypotheses == original.hypotheses
        assert restored.rule_index == original.rule_index

        for name in (
            "demonstrations",
            "query_input",
            "query_output",
            "valid_masks",
            "soft_targets",
        ):
            a = getattr(original, name)
            b = getattr(restored, name)
            assert a.dtype == b.dtype, name
            assert torch.equal(a, b), name


def test_sample_types_and_shapes(train):
    for index in range(len(train)):
        sample = train[index]
        num_demos = sample.demonstrations.shape[0]
        num_candidates, length = sample.op_ids.shape

        assert 2 <= num_demos <= 5
        assert sample.episode_index == index
        assert type(sample.rule_index) is int
        assert 0 <= sample.rule_index < num_candidates

        expected = {
            "demonstrations": ((num_demos, 2, 5, 5), torch.long),
            "query_input": ((5, 5), torch.long),
            "query_output": ((5, 5), torch.long),
            "valid_masks": ((num_demos, num_candidates), torch.bool),
            "soft_targets": ((num_demos, num_candidates), torch.float32),
            "op_ids": ((num_candidates, length), torch.long),
            "op_params": (
                (num_candidates, length, PARAM_CAPACITY),
                torch.long,
            ),
        }

        for name, (shape, dtype) in expected.items():
            value = getattr(sample, name)
            assert tuple(value.shape) == shape, name
            assert value.dtype == dtype, name


def test_dataset_covers_variable_sizes(train):
    sizes = {
        (s.demonstrations.shape[0], *s.op_ids.shape)
        for s in (train[i] for i in range(len(train)))
    }

    assert {t for t, _, _ in sizes} == {2, 3, 4, 5}
    assert len({k for _, k, _ in sizes}) > 1
    assert len({length for _, _, length in sizes}) > 1


def test_samples_do_not_alias_storage(train):
    sample = train[0]

    for field in fields(EpisodeSample):
        value = getattr(sample, field.name)

        if isinstance(value, torch.Tensor):
            value.zero_()

    fresh = train[0]

    assert fresh.valid_masks.any()
    assert fresh.soft_targets.sum() > 0
    assert (fresh.op_ids != PAD_OP_ID).any()


def test_index_handling(train):
    assert train[-1].episode_index == len(train) - 1

    for index in (len(train), -len(train) - 1):
        with pytest.raises(IndexError):
            train[index]

    for index in ("0", 1.5, slice(0, 2)):
        with pytest.raises(TypeError):
            train[index]


def test_collate_mixed_batch_shapes(train):
    by_length = {
        train[i].demonstrations.shape[0]: i for i in range(len(train))
    }
    samples = [train[by_length[2]], train[by_length[5]]]
    batch = collate_episodes(samples)

    t_max = 5
    k_max = max(s.op_ids.shape[0] for s in samples)
    l_max = max(s.op_ids.shape[1] for s in samples)

    expected = {
        "demonstrations": ((2, t_max, 2, 5, 5), torch.long),
        "demo_mask": ((2, t_max), torch.bool),
        "query_input": ((2, 5, 5), torch.long),
        "query_output": ((2, 5, 5), torch.long),
        "rule_index": ((2,), torch.long),
        "valid_masks": ((2, t_max, k_max), torch.bool),
        "soft_targets": ((2, t_max, k_max), torch.float32),
        "candidate_mask": ((2, k_max), torch.bool),
        "op_ids": ((2, k_max, l_max), torch.long),
        "op_params": ((2, k_max, l_max, PARAM_CAPACITY), torch.long),
        "op_mask": ((2, k_max, l_max), torch.bool),
        "param_mask": ((2, k_max, l_max, PARAM_CAPACITY), torch.bool),
        "episode_index": ((2,), torch.long),
    }

    assert {field.name for field in fields(EpisodeBatch)} == set(expected)

    for name, (shape, dtype) in expected.items():
        value = getattr(batch, name)
        assert tuple(value.shape) == shape, name
        assert value.dtype == dtype, name

    assert batch.batch_size == 2
    assert batch.num_demos.tolist() == [2, 5]
    assert batch.num_candidates.tolist() == [
        s.op_ids.shape[0] for s in samples
    ]


def test_padding_preserves_valid_data(train, full_batch):
    batch = full_batch

    for b in range(batch.batch_size):
        sample = train[b]
        t = sample.demonstrations.shape[0]
        k, length = sample.op_ids.shape
        op_mask, param_mask = operation_masks(sample.op_ids)

        # Real entries are copied unchanged.
        assert torch.equal(batch.demonstrations[b, :t], sample.demonstrations)
        assert torch.equal(batch.query_input[b], sample.query_input)
        assert torch.equal(batch.query_output[b], sample.query_output)
        assert batch.rule_index[b].item() == sample.rule_index
        assert batch.episode_index[b].item() == sample.episode_index
        assert torch.equal(batch.valid_masks[b, :t, :k], sample.valid_masks)
        assert torch.equal(batch.soft_targets[b, :t, :k], sample.soft_targets)
        assert torch.equal(batch.op_ids[b, :k, :length], sample.op_ids)
        assert torch.equal(batch.op_params[b, :k, :length], sample.op_params)
        assert torch.equal(batch.op_mask[b, :k, :length], op_mask)
        assert torch.equal(batch.param_mask[b, :k, :length], param_mask)

        # Masks mark exactly the real positions.
        assert batch.demo_mask[b].tolist() == (
            [True] * t + [False] * (batch.demo_mask.shape[1] - t)
        )
        assert batch.candidate_mask[b].tolist() == (
            [True] * k + [False] * (batch.candidate_mask.shape[1] - k)
        )

        # Padding holds the documented values and is never marked real.
        assert (batch.demonstrations[b, t:] == 0).all()
        assert not batch.valid_masks[b, t:].any()
        assert not batch.valid_masks[b, :, k:].any()
        assert (batch.soft_targets[b, t:] == 0).all()
        assert (batch.soft_targets[b, :, k:] == 0).all()
        assert (batch.op_ids[b, k:] == PAD_OP_ID).all()
        assert (batch.op_ids[b, :, length:] == PAD_OP_ID).all()
        assert (batch.op_params[b, k:] == PAD_PARAM_VALUE).all()
        assert (batch.op_params[b, :, length:] == PAD_PARAM_VALUE).all()
        assert not batch.op_mask[b, k:].any()
        assert not batch.param_mask[b, k:].any()

        # The batch still decodes to the episode's own programs.
        assert decode_hypotheses(
            batch.op_ids[b, :k], batch.op_params[b, :k]
        ) == train.episode(b).hypotheses


def test_targets_are_consistent_with_masks(full_batch):
    batch = full_batch
    real_steps = batch.demo_mask
    real_cells = real_steps[:, :, None] & batch.candidate_mask[:, None, :]

    totals = batch.soft_targets.sum(dim=-1)

    assert torch.allclose(
        totals[real_steps],
        torch.ones_like(totals[real_steps]),
    )
    assert (totals[~real_steps] == 0).all()
    assert not (batch.valid_masks & ~real_cells).any()
    assert (batch.soft_targets[~batch.valid_masks] == 0).all()

    # The ground-truth program is a real candidate and is valid at
    # exactly the real steps.
    assert (batch.rule_index < batch.num_candidates).all()

    truth = batch.valid_masks[
        torch.arange(batch.batch_size), :, batch.rule_index
    ]
    assert torch.equal(truth, real_steps)

    # Version spaces only shrink across real demonstrations.
    shrinking = batch.valid_masks[:, 1:] & ~batch.valid_masks[:, :-1]
    assert not shrinking.any()


def _toy_log_probs(batch: EpisodeBatch) -> torch.Tensor:
    """
    A small masked scorer standing in for A06/A07: [B, T, K].

    It embeds every model input, accumulates demonstrations causally,
    and normalizes only over real candidates.
    """

    generator = torch.Generator().manual_seed(0)

    # float64 keeps batched and unbatched results equal to ~1e-12.
    def embedding(rows):
        return torch.randn(
            rows, 8, generator=generator, dtype=torch.float64
        )

    grid_embedding = embedding(NUM_COLORS)
    op_embedding = embedding(NUM_OP_IDS)
    param_embedding = embedding(NUM_PARAM_VALUES)
    slot_embedding = embedding(PARAM_CAPACITY)

    demos = grid_embedding[batch.demonstrations].sum(dim=(2, 3, 4))
    demos = demos * batch.demo_mask[..., None]
    states = demos.cumsum(dim=1)

    params = param_embedding[batch.op_params] * slot_embedding
    params = (params * batch.param_mask[..., None]).sum(dim=3)
    ops = op_embedding[batch.op_ids] + params
    programs = (ops * batch.op_mask[..., None]).sum(dim=2)

    logits = torch.einsum("bte,bke->btk", states, programs)
    logits = logits.masked_fill(~batch.candidate_mask[:, None, :], -torch.inf)

    return torch.log_softmax(logits, dim=-1)


def _fill_padding(batch: EpisodeBatch) -> EpisodeBatch:
    """
    Overwrite every padded model input with arbitrary valid values.

    Real program slots are recomputed from the IDs rather than read
    from the batch's own masks, so a wrong mask cannot hide padding.
    """

    generator = torch.Generator().manual_seed(1)

    def noise(value, high):
        return torch.randint(high, value.shape, generator=generator)

    demo_real = batch.demo_mask[:, :, None, None, None].expand_as(
        batch.demonstrations
    )
    op_real, param_real = operation_masks(batch.op_ids)

    return replace(
        batch,
        demonstrations=torch.where(
            demo_real,
            batch.demonstrations,
            noise(batch.demonstrations, NUM_COLORS),
        ),
        op_ids=torch.where(
            op_real,
            batch.op_ids,
            noise(batch.op_ids, NUM_OP_IDS),
        ),
        op_params=torch.where(
            param_real,
            batch.op_params,
            noise(batch.op_params, NUM_PARAM_VALUES),
        ),
    )


def test_padding_does_not_affect_predictions(train, full_batch):
    batched = _toy_log_probs(full_batch)
    filled = _toy_log_probs(_fill_padding(full_batch))

    for b in range(full_batch.batch_size):
        t = full_batch.num_demos[b].item()
        k = full_batch.num_candidates[b].item()
        alone = _toy_log_probs(collate_episodes([train[b]]))[0]

        assert torch.isfinite(batched[b, :t, :k]).all()
        assert torch.allclose(batched[b, :t, :k], alone[:t, :k])
        assert torch.allclose(filled[b, :t, :k], alone[:t, :k])


def test_episodes_without_operations_collate():
    episode = ARCEpisode(
        demonstrations=torch.zeros((2, 2, 5, 5), dtype=torch.long),
        query_input=torch.zeros((5, 5), dtype=torch.long),
        query_output=torch.zeros((5, 5), dtype=torch.long),
        rule_index=0,
        valid_masks=torch.ones((2, 1), dtype=torch.bool),
        soft_targets=torch.ones((2, 1), dtype=torch.float32),
        hypotheses=(Hypothesis(),),
    )

    dataset = ARCEpisodeDataset([episode])
    batch = collate_episodes([dataset[0]])

    assert batch.op_ids.shape == (1, 1, 1)
    assert not batch.op_mask.any()
    assert batch.candidate_mask.tolist() == [[True]]
    assert dataset.episode(0).hypotheses == (Hypothesis(),)


def test_collate_rejects_invalid_samples(train):
    with pytest.raises(ValueError, match="empty"):
        collate_episodes([])

    with pytest.raises(TypeError):
        collate_episodes([train[0].demonstrations])

    sample = train[0]
    inconsistent = replace(sample, valid_masks=sample.valid_masks[:, :-1])

    with pytest.raises(ValueError, match="valid_masks"):
        collate_episodes([inconsistent])


def test_dataset_rejects_non_episodes():
    with pytest.raises(TypeError):
        ARCEpisodeDataset([object()])


def test_dataloader_iterates_every_episode(train):
    loader = make_dataloader(train, batch_size=5, shuffle=True, seed=0)
    batches = list(loader)

    assert all(isinstance(batch, EpisodeBatch) for batch in batches)
    assert [batch.batch_size for batch in batches] == [5, 5, 5, 5, 4]

    seen = torch.cat([batch.episode_index for batch in batches])
    assert sorted(seen.tolist()) == list(range(len(train)))

    dropped = make_dataloader(
        train, batch_size=5, shuffle=True, seed=0, drop_last=True
    )
    assert [batch.batch_size for batch in dropped] == [5, 5, 5, 5]


def test_unshuffled_loader_preserves_order(train):
    loader = make_dataloader(train, batch_size=7)
    seen = torch.cat([batch.episode_index for batch in loader])

    assert seen.tolist() == list(range(len(train)))


def test_shuffling_is_reproducible(train):
    def two_epochs(seed):
        loader = make_dataloader(train, batch_size=8, shuffle=True, seed=seed)
        torch.manual_seed(1234)  # Global RNG state must not matter.
        first = list(loader)
        torch.rand(10)
        return first, list(loader)

    first_run = two_epochs(seed=0)
    second_run = two_epochs(seed=0)

    for epoch_a, epoch_b in zip(first_run, second_run):
        assert len(epoch_a) == len(epoch_b)

        for batch_a, batch_b in zip(epoch_a, epoch_b):
            assert_batches_equal(batch_a, batch_b)

    def order(epoch):
        return torch.cat([batch.episode_index for batch in epoch]).tolist()

    assert order(first_run[0]) != order(first_run[1])
    assert order(first_run[0]) != order(two_epochs(seed=1)[0])


def test_loading_is_reproducible(snapshot, full_batch):
    reloaded = ARCEpisodeDataset.from_snapshot(snapshot, "train")
    batch = collate_episodes([reloaded[i] for i in range(len(reloaded))])

    assert_batches_equal(batch, full_batch)


def test_workers_match_single_process(train):
    def batches(num_workers):
        loader = make_dataloader(
            train,
            batch_size=6,
            shuffle=True,
            seed=3,
            num_workers=num_workers,
        )
        return list(loader)

    single = batches(num_workers=0)
    parallel = batches(num_workers=2)

    assert len(single) == len(parallel)

    for batch_a, batch_b in zip(single, parallel):
        assert_batches_equal(batch_a, batch_b)


def _devices():
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    if torch.backends.mps.is_available():
        devices.append("mps")

    return devices


@pytest.mark.parametrize("device", _devices())
def test_batch_moves_to_device(full_batch, device):
    moved = full_batch.to(device)

    assert isinstance(moved, EpisodeBatch)

    for field in fields(EpisodeBatch):
        value = getattr(moved, field.name)
        assert value.device.type == device, field.name

    assert_batches_equal(moved.to("cpu"), full_batch)


def test_make_dataloader_validates_arguments(train):
    with pytest.raises(TypeError):
        make_dataloader([train[0]], batch_size=1)

    for batch_size in (0, -1, 2.0):
        with pytest.raises(ValueError):
            make_dataloader(train, batch_size=batch_size)

    with pytest.raises(TypeError):
        make_dataloader(train, batch_size=1, seed="0")


def test_snapshot_integrity_is_checked(tmp_path):
    directory = tmp_path / "v1"
    save_dataset(
        directory,
        GenerationConfig(train_size=3, validation_size=1, test_size=0),
    )

    with pytest.raises(ValueError, match="Unknown split"):
        ARCEpisodeDataset.from_snapshot(directory, "dev")

    empty = ARCEpisodeDataset.from_snapshot(directory, "test")
    assert len(empty) == 0
    assert list(make_dataloader(empty, batch_size=4)) == []

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["validation"]["count"] = 2
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest lists 2"):
        ARCEpisodeDataset.from_snapshot(directory, "validation")

    with (directory / "train.jsonl.gz").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="Checksum"):
        ARCEpisodeDataset.from_snapshot(directory, "train")
