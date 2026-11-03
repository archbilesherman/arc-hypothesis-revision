import pytest
import torch

from arc_revision.data.schema import ARCEpisode
from arc_revision.data.hypotheses import Hypothesis, Operation



def make_valid_episode(num_demos: int = 2) -> ARCEpisode:
    """Create a structurally valid episode."""

    hypotheses = (
        Hypothesis(),
        Hypothesis((Operation("rotate", (90,)),)),
        Hypothesis((Operation("replace_color", (1, 9)),)),
    )

    masks = torch.tensor(
        [[True, True, True]]
        + [[True, False, True]] * (num_demos - 1),
        dtype=torch.bool,
    )

    targets = (
        masks.float()
        / masks.sum(dim=-1, keepdim=True)
    )

    return ARCEpisode(
        demonstrations=torch.zeros(
            (num_demos, 2, 5, 5),
            dtype=torch.long,
        ),
        query_input=torch.zeros((5, 5), dtype=torch.long),
        query_output=torch.zeros((5, 5), dtype=torch.long),
        rule_index=0,
        valid_masks=masks,
        soft_targets=targets,
        hypotheses=hypotheses,
    )



def test_valid_episode():
    episode = make_valid_episode()
    assert episode.demonstrations.shape == (2, 2, 5, 5)
    assert episode.rule_index == 0


def test_invalid_grid_shape():
    episode = make_valid_episode()
    episode.query_input = torch.zeros((4, 5), dtype=torch.long)

    with pytest.raises(ValueError):
        episode.validate()


@pytest.mark.parametrize("color", [-1, 10, 15])
def test_invalid_colors(color):
    episode = make_valid_episode()
    episode.query_input[0, 0] = color

    with pytest.raises(ValueError):
        episode.validate()


@pytest.mark.parametrize("index", [-1, 3, 5])
def test_invalid_rule_index(index):
    episode = make_valid_episode()
    episode.rule_index = index

    with pytest.raises(ValueError):
        episode.validate()


@pytest.mark.parametrize("num_demos", [2, 3, 4, 5])
def test_supported_lengths(num_demos):
    episode = make_valid_episode(num_demos)

    assert episode.valid_masks.shape == (num_demos, 3)


@pytest.mark.parametrize("num_demos", [1, 6])
def test_invalid_demo_count(num_demos):
    with pytest.raises(ValueError, match="demonstrations"):
        make_valid_episode(num_demos)


def test_wrong_candidate_length():
    episode = make_valid_episode()
    episode.valid_masks = torch.ones(
        (2, 8),
        dtype=torch.bool,
    )

    with pytest.raises(ValueError):
        episode.validate()


def test_eliminated_rule_cannot_return():
    episode = make_valid_episode(3)
    episode.valid_masks[-1, 1] = True

    with pytest.raises(
        ValueError,
        match="Previously eliminated hypotheses",
    ):
        episode.validate()


def test_invalid_grid_dtype():
    episode = make_valid_episode()
    episode.query_input = episode.query_input.float()

    with pytest.raises(TypeError):
        episode.validate()


def test_invalid_mask_shape():
    episode = make_valid_episode()
    episode.valid_masks = torch.ones((2, 7), dtype=torch.bool)

    with pytest.raises(ValueError):
        episode.validate()


def test_invalid_target_distribution():
    episode = make_valid_episode()
    episode.soft_targets[0] = 0

    with pytest.raises(ValueError):
        episode.validate()


def test_probability_on_invalid_rule():
    episode = make_valid_episode()

    episode.soft_targets[1, 0] = 0.4
    episode.soft_targets[1, 1] = 0.1

    with pytest.raises(ValueError):
        episode.validate()


def test_ground_truth_eliminated():
    episode = make_valid_episode()
    episode.valid_masks[1, 0] = False

    with pytest.raises(ValueError):
        episode.validate()


def test_eliminated_rule_reappears():
    episode = make_valid_episode(3)

    episode.valid_masks[2, 1] = True

    with pytest.raises(
        ValueError,
        match="Previously eliminated hypotheses",
    ):
        episode.validate()


def test_nonfinite_targets():
    episode = make_valid_episode()
    episode.soft_targets[0, 0] = float("nan")

    with pytest.raises(ValueError):
        episode.validate()