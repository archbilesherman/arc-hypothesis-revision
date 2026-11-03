import pytest
import torch

from arc_revision.models.encoder import SharedCNNGridEncoder


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
