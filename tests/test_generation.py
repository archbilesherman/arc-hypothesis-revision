"""Tests for synthetic ARC episode generation and persistence."""

import json
from collections import Counter

import pytest
import torch

from arc_revision.data.generator import (
    GenerationConfig,
    generate_episode,
    iter_split,
    load_split,
    save_dataset,
)
from arc_revision.data.hypotheses import apply_hypothesis
from arc_revision.data.version_space import (
    consistent_hypotheses,
    compute_soft_targets,
    compute_version_spaces,
)


@pytest.mark.parametrize("num_demos", [2, 3, 4, 5])
@pytest.mark.parametrize("ambiguous", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_generation(num_demos, ambiguous, seed):
    episode = generate_episode(
        seed=seed,
        num_demos=num_demos,
        ambiguous=ambiguous,
    )

    assert episode.demonstrations.shape == (
        num_demos, 2, 5, 5
    )

    assert len(episode.hypotheses) == 2 ** (num_demos + 1)

    expected = (
        [
            2 ** (num_demos - i - 1)
            for i in range(num_demos)
        ]
        if ambiguous
        else [1] * num_demos
    )

    assert episode.valid_masks.sum(1).tolist() == expected

    true_program = episode.hypotheses[episode.rule_index]

    for x, y in episode.demonstrations:
        assert torch.equal(
            apply_hypothesis(x, true_program), y
        )

        if ambiguous:
            assert consistent_hypotheses(
                x, y, episode.hypotheses
            ).sum().item() > 1

    assert torch.equal(
        apply_hypothesis(episode.query_input, true_program),
        episode.query_output,
    )

    assert torch.equal(
        compute_version_spaces(
            episode.demonstrations,
            episode.hypotheses,
        ),
        episode.valid_masks,
    )

    assert torch.allclose(
        compute_soft_targets(episode.valid_masks),
        episode.soft_targets,
    )


def test_repeatability():
    a = generate_episode(seed=42, num_demos=5)
    b = generate_episode(seed=42, num_demos=5)

    assert a.hypotheses == b.hypotheses
    assert a.rule_index == b.rule_index
    assert torch.equal(a.demonstrations, b.demonstrations)


def test_split_distribution():
    config = GenerationConfig(
        train_size=40,
        validation_size=8,
        test_size=8,
    )

    episodes = list(iter_split("train", config))

    assert len(episodes) == 40

    assert Counter(
        ep.demonstrations.shape[0]
        for ep in episodes
    ) == {2: 10, 3: 10, 4: 10, 5: 10}

    assert sum(
        ep.valid_masks[0].sum().item() > 1
        for ep in episodes
    ) == 24


def test_dataset_save_and_load(tmp_path):
    config = GenerationConfig(
        train_size=12,
        validation_size=4,
        test_size=4,
    )

    directory = tmp_path / "v1"
    manifest_path = save_dataset(directory, config)

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text()
    )

    assert manifest["splits"]["train"]["count"] == 12

    expected = list(iter_split("train", config))
    loaded = list(load_split(directory, "train"))

    assert len(loaded) == len(expected)

    for original, restored in zip(expected, loaded):
        assert torch.equal(
            original.demonstrations,
            restored.demonstrations,
        )
        assert original.hypotheses == restored.hypotheses
        assert original.rule_index == restored.rule_index
        assert torch.equal(
            original.valid_masks,
            restored.valid_masks,
        )

    with pytest.raises(FileExistsError):
        save_dataset(directory, config)


def test_snapshot_detects_corruption(tmp_path):
    config = GenerationConfig(
        train_size=2,
        validation_size=0,
        test_size=0,
    )

    directory = tmp_path / "v1"
    save_dataset(directory, config)

    with (directory / "train.jsonl.gz").open("ab") as f:
        f.write(b"invalid")

    with pytest.raises(ValueError, match="Checksum"):
        list(load_split(directory, "train"))


def test_bounded_retries(monkeypatch):
    import arc_revision.data.generator as generator

    monkeypatch.setattr(
        generator,
        "_sample_grid",
        lambda rng, markers, anchors: torch.zeros(
            (5, 5), dtype=torch.long
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="after 2 attempts",
    ):
        generate_episode(
            seed=1,
            num_demos=5,
            ambiguous=True,
            max_attempts=2,
        )


@pytest.mark.parametrize("num_demos", [1, 6])
def test_invalid_demo_count(num_demos):
    with pytest.raises(ValueError):
        generate_episode(seed=1, num_demos=num_demos)


def test_invalid_attempts():
    with pytest.raises(ValueError):
        generate_episode(seed=1, max_attempts=0)