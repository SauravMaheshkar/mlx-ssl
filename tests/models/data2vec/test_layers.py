import mlx.core as mx
import pytest

from src.models.data2vec.layers import PatchEncoder, VisionEmbeddings


@pytest.mark.parametrize(
    "image_size, patch_size, num_channels, hidden_size, seq_len, patch_height, patch_width",  # noqa: E501
    [
        (224, 16, 3, 768, 196, 14, 14),
        (256, 32, 3, 512, 64, 8, 8),
    ],
)
def test_patch_encoder(
    image_size, patch_size, num_channels, hidden_size, seq_len, patch_height, patch_width
):
    encoder = PatchEncoder(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
    )

    pixel_values = mx.ones((1, image_size, image_size, num_channels))
    embeddings, (height, width) = encoder(pixel_values)
    assert embeddings.shape == (1, seq_len, hidden_size)
    assert height == patch_height
    assert width == patch_width


@pytest.mark.parametrize(
    "image_size, patch_size, num_channels, hidden_size, seq_len, patch_height, patch_width",  # noqa: E501
    [
        (224, 16, 3, 768, 196, 14, 14),
        (256, 32, 3, 512, 64, 8, 8),
    ],
)
def test_vision_embeddings(
    image_size, patch_size, num_channels, hidden_size, seq_len, patch_height, patch_width
):
    module = VisionEmbeddings(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
    )

    pixel_values = mx.ones((1, image_size, image_size, num_channels))
    embeddings, (height, width) = module(pixel_values)
    assert embeddings.shape == (1, seq_len + 1, hidden_size)
    assert height == patch_height
    assert width == patch_width
