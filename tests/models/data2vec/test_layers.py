import mlx.core as mx
import pytest

from src.models.data2vec.layers import PatchEncoder, VisionEmbeddings, VisionSelfAttention


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


@pytest.mark.parametrize(
    "hidden_size, num_attention_heads, seq_len, batch_size , image_size, num_channels, patch_size",  # noqa: E501
    [
        (768, 12, 196, 1, 224, 3, 16),
        (512, 8, 64, 1, 256, 3, 32),
    ],
)
def test_vision_self_attention(
    hidden_size,
    num_attention_heads,
    seq_len,
    batch_size,
    image_size,
    num_channels,
    patch_size,
):
    attn_module = VisionSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    encoder = PatchEncoder(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
    )

    pixel_values = mx.ones((batch_size, image_size, image_size, num_channels))
    hidden_states, _ = encoder(pixel_values)

    context_layer = attn_module(hidden_states)
    assert context_layer.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.parametrize(
    "hidden_size, num_attention_heads, seq_len, batch_size , image_size, num_channels, patch_size",  # noqa: E501
    [
        (768, 12, 196, 1, 224, 3, 16),
        (512, 8, 64, 1, 256, 3, 32),
    ],
)
def test_vision_self_attention_with_attn_output(
    hidden_size,
    num_attention_heads,
    seq_len,
    batch_size,
    image_size,
    num_channels,
    patch_size,
):
    attn_module = VisionSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    encoder = PatchEncoder(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
    )

    pixel_values = mx.ones((batch_size, image_size, image_size, num_channels))
    hidden_states, _ = encoder(pixel_values)

    context_layer, attn_probs = attn_module(hidden_states, output_attns=True)
    assert context_layer.shape == (batch_size, seq_len, hidden_size)
    assert attn_probs.shape == (batch_size, num_attention_heads, seq_len, seq_len)
