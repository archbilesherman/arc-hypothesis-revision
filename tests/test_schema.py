
import pytest
import torch

from arc_revision.data.schema import ARCEpisode


def make_valid_episode() -> ARCEpisode:
    """Create a valid synthetic episode for testing."""

    return ARCEpisode(
        demonstrations=torch.zeros((2, 2, 5, 5), dtype=torch.long),
        query_input=torch.zeros((5, 5), dtype=torch.long),
        query_output=torch.zeros((5, 5), dtype=torch.long),
        rule_index=0,
        valid_masks=torch.tensor([
            [True, True, True, False, False, False, False, False],
            [True, False, True, False, False, False, False, False],
        ]),
        soft_targets=torch.tensor([
            [1/3, 1/3, 1/3, 0, 0, 0, 0, 0],
            [0.5, 0, 0.5, 0, 0, 0, 0, 0],
        ], dtype=torch.float32),
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


@pytest.mark.parametrize("index", [-1, 8, 10])
def test_invalid_rule_index(index):
    episode = make_valid_episode()
    episode.rule_index = index

    with pytest.raises(ValueError):
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
    episode.soft_targets[0, 0] = 0.2
    episode.soft_targets[0, 3] = 0.13333334

    with pytest.raises(ValueError):
        episode.validate()


def test_ground_truth_eliminated():
    episode = make_valid_episode()
    episode.valid_masks[1, 0] = False

    with pytest.raises(ValueError):
        episode.validate()


def test_eliminated_rule_reappears():
    episode = make_valid_episode()
    episode.valid_masks[1, 3] = True

    with pytest.raises(ValueError):
        episode.validate()


def test_nonfinite_targets():
    episode = make_valid_episode()
    episode.soft_targets[0, 0] = float("nan")

    with pytest.raises(ValueError):
        episode.validate()
