import pytest
import torch

from arc_revision.models.encoder import SharedCNNGridEncoder
from arc_revision.data.dataset import (
    ARCEpisodeDataset,
    collate_episodes,
)
from arc_revision.data.generator import (
    GenerationConfig,
    iter_split,
)

@pytest.mark.parametrize(
    "batch_size,steps,embedding_dim",
    [(1, 2, 32), (3, 5, 128), (4, 3, 64)],
)
def test_output_shape(batch_size, steps, embedding_dim):
    model = SharedCNNGridEncoder(embedding_dim=embedding_dim)
    demonstrations = torch.randint(
        0, 10, (batch_size, steps, 2, 5, 5)
    )

    embeddings = model(demonstrations)

    assert embeddings.shape == (
        batch_size, steps, embedding_dim
    )
    assert torch.isfinite(embeddings).all()


def test_shared_cnn_processes_both_grids():
    model = SharedCNNGridEncoder()
    received_shapes = []

    hook = model.grid_encoder.register_forward_pre_hook(
        lambda module, args: received_shapes.append(args[0].shape)
    )

    try:
        model(torch.randint(0, 10, (2, 3, 2, 5, 5)))
    finally:
        hook.remove()

    assert received_shapes == [
        torch.Size((12, 10, 5, 5))
    ]


def test_masked_padding_is_zero_and_cannot_affect_valid_embeddings():
    torch.manual_seed(5)
    model = SharedCNNGridEncoder(embedding_dim=48)

    demonstrations = torch.randint(
        0, 10, (2, 4, 2, 5, 5)
    )

    mask = torch.tensor([
        [True, True, False, False],
        [True, False, False, False],
    ])

    original = model(demonstrations, mask)

    modified = demonstrations.clone()
    modified[~mask] = -1

    after_change = model(modified, mask)

    assert torch.all(original[~mask] == 0)
    assert torch.allclose(original, after_change)
    assert torch.all(torch.isfinite(original))


def test_gradients_reach_cnn_and_pair_projection():
    model = SharedCNNGridEncoder()
    demonstrations = torch.randint(
        0, 10, (2, 3, 2, 5, 5)
    )

    loss = model(demonstrations).square().mean()
    loss.backward()

    assert all(
        param.grad is not None
        for param in model.parameters()
    )

    assert all(
        torch.isfinite(param.grad).all()
        for param in model.parameters()
    )

    assert any(
        torch.any(param.grad != 0)
        for param in model.parameters()
    )


@pytest.mark.parametrize(
    "demonstrations,mask,error",
    [
        (
            torch.zeros(2, 3, 5, 5, dtype=torch.long),
            None,
            ValueError,
        ),
        (
            torch.zeros(2, 3, 2, 6, 5, dtype=torch.long),
            None,
            ValueError,
        ),
        (
            torch.zeros(2, 3, 2, 5, 5, dtype=torch.float32),
            None,
            TypeError,
        ),
        (
            torch.full((2, 3, 2, 5, 5), 10, dtype=torch.long),
            None,
            ValueError,
        ),
        (
            torch.zeros(2, 3, 2, 5, 5, dtype=torch.long),
            torch.ones(2, 2, dtype=torch.bool),
            ValueError,
        ),
    ],
)
def test_invalid_inputs(demonstrations, mask, error):
    with pytest.raises(error):
        SharedCNNGridEncoder()(demonstrations, mask)


@pytest.fixture
def mixed_episode_batch():
    config = GenerationConfig(
        train_size=4,
        validation_size=0,
        test_size=0,
        ambiguous_fraction=0.0,
        seed=42,
    )

    dataset = ARCEpisodeDataset(iter_split("train", config))

    by_length = {
        dataset[i].demonstrations.shape[0]: dataset[i]
        for i in range(len(dataset))
    }

    samples = [by_length[2], by_length[5]]
    batch = collate_episodes(samples)

    return samples, batch

def test_dataset_batch_integration(mixed_episode_batch):
    _, batch = mixed_episode_batch

    encoder = SharedCNNGridEncoder(embedding_dim=128)
    embeddings = encoder(
        batch.demonstrations,
        demonstration_mask=batch.demo_mask,
    )

    assert batch.num_demos.tolist() == [2, 5]
    assert embeddings.shape == (2, 5, 128)
    assert torch.isfinite(embeddings).all()
    assert torch.all(embeddings[~batch.demo_mask] == 0)

def test_dataset_padding_independence(mixed_episode_batch):
    _, batch = mixed_episode_batch

    encoder = SharedCNNGridEncoder(embedding_dim=128)
    encoder.eval()

    modified = batch.demonstrations.clone()
    modified[~batch.demo_mask] = -1

    with torch.no_grad():
        original = encoder(
            batch.demonstrations,
            batch.demo_mask,
        )
        changed = encoder(
            modified,
            batch.demo_mask,
        )

    torch.testing.assert_close(original, changed)
    assert torch.all(changed[~batch.demo_mask] == 0)

def test_individual_vs_batched_encoding(mixed_episode_batch):
    samples, batch = mixed_episode_batch

    encoder = SharedCNNGridEncoder(embedding_dim=128)
    encoder.eval()

    with torch.no_grad():
        batched = encoder(
            batch.demonstrations,
            batch.demo_mask,
        )

        for i, sample in enumerate(samples):
            single = encoder(
                sample.demonstrations.unsqueeze(0)
            )

            num_demos = sample.demonstrations.shape[0]

            torch.testing.assert_close(
                batched[i, :num_demos],
                single.squeeze(0),
            )