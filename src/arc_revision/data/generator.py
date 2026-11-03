"""Seeded ARC episodes with controlled hypothesis ambiguity."""

import gzip
import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch

from arc_revision.data.hypotheses import (
    Hypothesis,
    Operation,
    apply_hypothesis,
)
from arc_revision.data.schema import (
    ARCEpisode,
    MAX_DEMOS,
    MIN_DEMOS,
)
from arc_revision.data.version_space import (
    consistent_hypotheses,
    compute_soft_targets,
    compute_version_spaces,
)


DATASET_VERSION = 1

SPLIT_OFFSETS = {
    "train": 0,
    "validation": 1_000_003,
    "test": 2_000_003,
}


def _geometry_program(index: int) -> tuple[Operation, ...]:
    """Represent a D4 operation using the hypothesis DSL."""
    operations = (
        (),
        (Operation("rotate", (90,)),),
        (Operation("rotate", (180,)),),
        (Operation("rotate", (270,)),),
        (Operation("reflect_horizontal"),),
        (Operation("reflect_vertical"),),
        (Operation("reflect_main_diagonal"),),
        (Operation("reflect_anti_diagonal"),),
    )
    return operations[index]


def _candidate_space(rng: random.Random, num_demos: int):
    """Build candidate programs with independent color choices."""
    markers = tuple(rng.sample(range(1, 10), num_demos))
    targets = [
        color for color in range(1, 10)
        if color not in markers
    ]

    choices = [
        tuple(rng.sample(targets, 2))
        for _ in markers
    ]

    geometries = rng.sample(range(8), 2)
    true_geometry = rng.randrange(2)
    true_bits = rng.randrange(1 << num_demos)

    candidates = []

    for geometry in geometries:
        for bits in range(1 << num_demos):
            replacements = tuple(
                Operation(
                    "replace_color",
                    (marker, choices[i][(bits >> i) & 1]),
                )
                for i, marker in enumerate(markers)
            )

            candidates.append(
                Hypothesis(
                    _geometry_program(geometry) + replacements
                )
            )

    true_program = candidates[
        true_geometry * (1 << num_demos) + true_bits
    ]

    rng.shuffle(candidates)
    candidates = tuple(candidates)

    return (
        candidates,
        candidates.index(true_program),
        markers,
        tuple(targets),
    )


def _sample_grid(
    rng: random.Random,
    markers: tuple[int, ...],
    anchor_colors: tuple[int, ...],
) -> torch.Tensor:
    """Generate a grid containing the requested marker colors."""
    palette = (0, 0, 0, 0, *anchor_colors)
    values = [rng.choice(palette) for _ in range(25)]

    positions = rng.sample(range(25), len(markers))

    for position, marker in zip(positions, markers):
        values[position] = marker

    return torch.tensor(
        values,
        dtype=torch.long,
    ).reshape(5, 5)


def generate_episode(
    *,
    seed: int,
    num_demos: int = 2,
    ambiguous: bool = True,
    max_attempts: int = 100,
) -> ARCEpisode:
    """Generate one identifiable synthetic ARC episode."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    if (
        type(num_demos) is not int
        or not MIN_DEMOS <= num_demos <= MAX_DEMOS
    ):
        raise ValueError(
            f"num_demos must be {MIN_DEMOS}-{MAX_DEMOS}"
        )

    if type(ambiguous) is not bool:
        raise TypeError("ambiguous must be a boolean")

    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    rng = random.Random(seed)

    candidates, true_index, markers, anchors = (
        _candidate_space(rng, num_demos)
    )

    true_program = candidates[true_index]

    for _ in range(max_attempts):
        pairs = []

        for step in range(num_demos):
            if ambiguous:
                present = (markers[step],)
            elif step == 0:
                present = markers
            else:
                present = tuple(
                    rng.sample(
                        markers,
                        rng.randint(1, num_demos),
                    )
                )

            grid = _sample_grid(rng, present, anchors)

            pairs.append(
                torch.stack((
                    grid,
                    apply_hypothesis(grid, true_program),
                ))
            )

        demonstrations = torch.stack(pairs)

        masks = compute_version_spaces(
            demonstrations,
            candidates,
        )

        sizes = masks.sum(dim=1).tolist()

        expected = (
            [
                1 << (num_demos - step - 1)
                for step in range(num_demos)
            ]
            if ambiguous
            else [1] * num_demos
        )

        if sizes != expected:
            continue

        if ambiguous and any(
            consistent_hypotheses(
                x, y, candidates
            ).sum().item() <= 1
            for x, y in demonstrations
        ):
            continue

        query_input = _sample_grid(
            rng, markers, anchors
        )
        query_output = apply_hypothesis(
            query_input, true_program
        )

        if any(
            torch.equal(query_input, pair[0])
            for pair in pairs
        ):
            continue

        if (
            consistent_hypotheses(
                query_input,
                query_output,
                candidates,
            ).sum().item() != 1
        ):
            continue

        return ARCEpisode(
            demonstrations=demonstrations,
            query_input=query_input,
            query_output=query_output,
            rule_index=true_index,
            valid_masks=masks,
            soft_targets=compute_soft_targets(masks),
            hypotheses=candidates,
        )

    raise RuntimeError(
        f"Could not generate {num_demos}-demo "
        f"{'ambiguous' if ambiguous else 'unambiguous'} "
        f"episode after {max_attempts} attempts"
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Reproducible dataset generation configuration."""

    train_size: int = 20_000
    validation_size: int = 2_000
    test_size: int = 2_000
    ambiguous_fraction: float = 0.60
    min_demos: int = MIN_DEMOS
    max_demos: int = MAX_DEMOS
    seed: int = 42
    max_attempts: int = 100


def iter_split(
    split: Literal["train", "validation", "test"],
    config: GenerationConfig = GenerationConfig(),
) -> Iterator[ARCEpisode]:
    """Stream episodes with the configured ambiguity ratio."""

    if split not in SPLIT_OFFSETS:
        raise ValueError("Unknown split")

    if any(
        type(n) is not int or n < 0
        for n in (
            config.train_size,
            config.validation_size,
            config.test_size,
        )
    ):
        raise ValueError(
            "Split sizes must be nonnegative integers"
        )

    if not 0 <= config.ambiguous_fraction <= 1:
        raise ValueError(
            "ambiguous_fraction must be between 0 and 1"
        )

    if not (
        MIN_DEMOS <= config.min_demos
        <= config.max_demos <= MAX_DEMOS
    ):
        raise ValueError("Invalid demonstration range")

    if type(config.seed) is not int:
        raise TypeError("seed must be an integer")

    size = getattr(config, f"{split}_size")
    ambiguous_count = round(
        size * config.ambiguous_fraction
    )

    rng = random.Random(
        config.seed + SPLIT_OFFSETS[split]
    )

    lengths = range(
        config.min_demos,
        config.max_demos + 1,
    )

    specs = [
        (
            ambiguous,
            config.min_demos + (i % len(lengths)),
        )
        for ambiguous, count in (
            (True, ambiguous_count),
            (False, size - ambiguous_count),
        )
        for i in range(count)
    ]

    rng.shuffle(specs)

    for ambiguous, num_demos in specs:
        yield generate_episode(
            seed=rng.getrandbits(64),
            num_demos=num_demos,
            ambiguous=ambiguous,
            max_attempts=config.max_attempts,
        )


def _encode(episode: ARCEpisode) -> dict:
    """Serialize an episode using portable JSON values."""
    return {
        "demonstrations": episode.demonstrations.tolist(),
        "query_input": episode.query_input.tolist(),
        "query_output": episode.query_output.tolist(),
        "rule_index": episode.rule_index,
        "valid_masks": episode.valid_masks.tolist(),
        "soft_targets": episode.soft_targets.tolist(),
        "hypotheses": [
            [
                [op.kind, list(op.params)]
                for op in h.operations
            ]
            for h in episode.hypotheses
        ],
    }


def _decode(record: dict) -> ARCEpisode:
    """Reconstruct an episode from saved JSON."""

    hypotheses = tuple(
        Hypothesis(tuple(
            Operation(kind, tuple(params))
            for kind, params in ops
        ))
        for ops in record["hypotheses"]
    )

    return ARCEpisode(
        demonstrations=torch.tensor(
            record["demonstrations"],
            dtype=torch.long,
        ),
        query_input=torch.tensor(
            record["query_input"],
            dtype=torch.long,
        ),
        query_output=torch.tensor(
            record["query_output"],
            dtype=torch.long,
        ),
        rule_index=record["rule_index"],
        valid_masks=torch.tensor(
            record["valid_masks"],
            dtype=torch.bool,
        ),
        soft_targets=torch.tensor(
            record["soft_targets"],
            dtype=torch.float32,
        ),
        hypotheses=hypotheses,
    )


def _sha256(path: Path) -> str:
    """Calculate a file checksum without loading it into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_dataset(
    directory: str | Path,
    config: GenerationConfig = GenerationConfig(),
) -> Path:
    """Save fixed dataset splits and a checksum manifest."""

    directory = Path(directory)

    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"{directory} is not empty; "
            "use a new dataset version directory"
        )

    directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": DATASET_VERSION,
        "config": vars(config),
        "splits": {},
    }

    for split in SPLIT_OFFSETS:
        path = directory / f"{split}.jsonl.gz"
        count = 0
        ambiguous_count = 0

        with path.open("wb") as binary:
            with gzip.GzipFile(
                filename="",
                fileobj=binary,
                mode="wb",
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="utf-8",
                    newline="\n",
                ) as output:
                    for episode in iter_split(split, config):
                        json.dump(
                            _encode(episode),
                            output,
                            separators=(",", ":"),
                        )
                        output.write("\n")

                        count += 1
                        ambiguous_count += int(
                            episode.valid_masks[0]
                            .sum().item() > 1
                        )

        manifest["splits"][split] = {
            "file": path.name,
            "count": count,
            "ambiguous_count": ambiguous_count,
            "sha256": _sha256(path),
        }

    manifest_path = directory / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    return manifest_path


def load_split(
    directory: str | Path,
    split: Literal["train", "validation", "test"],
) -> Iterator[ARCEpisode]:
    """Load fixed episodes from a saved dataset snapshot."""

    if split not in SPLIT_OFFSETS:
        raise ValueError("Unknown split")

    directory = Path(directory)

    manifest = json.loads(
        (directory / "manifest.json")
        .read_text(encoding="utf-8")
    )

    info = manifest["splits"][split]
    path = directory / info["file"]

    if _sha256(path) != info["sha256"]:
        raise ValueError(
            f"Checksum mismatch for {split}"
        )

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            yield _decode(json.loads(line))