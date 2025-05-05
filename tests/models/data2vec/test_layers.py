import mlx.core as mx
import pytest

from src.models.data2vec.layers import (
    Data2VecVisionBackbone,
    PatchEncoder,
    VisionAttention,
    VisionEmbeddings,
    VisionEncoder,
    VisionLayer,
    VisionSelfAttention,
)


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
def test_vision_attention(
    hidden_size,
    num_attention_heads,
    seq_len,
    batch_size,
    image_size,
    num_channels,
    patch_size,
):
    attn = VisionAttention(
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
    attn_out = attn(hidden_states)

    assert attn_out.shape == (seq_len, hidden_size)


@pytest.mark.parametrize(
    "hidden_size, intermediate_size, num_attention_heads, image_size, patch_size, num_channels, seq_len",  # noqa: E501
    [
        (768, 3072, 12, 224, 16, 3, 196),
        (512, 2048, 8, 256, 32, 3, 64),
    ],
)
def test_vision_layer(
    hidden_size,
    intermediate_size,
    num_attention_heads,
    image_size,
    patch_size,
    num_channels,
    seq_len,
):
    layer = VisionLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )

    encoder = PatchEncoder(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
    )

    pixel_values = mx.ones((1, image_size, image_size, num_channels))
    hidden_states, _ = encoder(pixel_values)
    layer_out = layer(hidden_states)

    assert layer_out.shape == (1, seq_len, hidden_size)


@pytest.mark.parametrize(
    "hidden_size, intermediate_size, num_attention_heads, image_size, patch_size, num_channels, seq_len",  # noqa: E501
    [
        (768, 3072, 12, 224, 16, 3, 196),
        (512, 2048, 8, 256, 32, 3, 64),
    ],
)
def test_vision_encoder(
    hidden_size,
    intermediate_size,
    num_attention_heads,
    image_size,
    patch_size,
    num_channels,
    seq_len,
):
    encoder = VisionEncoder(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=1,
    )

    encoder_module = PatchEncoder(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
    )

    pixel_values = mx.ones((1, image_size, image_size, num_channels))
    hidden_states, _ = encoder_module(pixel_values)
    encoder_out = encoder(hidden_states)

    assert encoder_out.shape == (1, seq_len, hidden_size)


@pytest.mark.parametrize(
    "image_size, patch_size, num_channels, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, seq_len",  # noqa: E501
    [
        (224, 16, 3, 768, 3072, 12, 12, 196),
        (256, 32, 3, 512, 2048, 8, 6, 64),
    ],
)
def test_data2vec_vision_backbone(
    image_size,
    patch_size,
    num_channels,
    hidden_size,
    intermediate_size,
    num_attention_heads,
    num_hidden_layers,
    seq_len,
):
    backbone = Data2VecVisionBackbone(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
    )

    pixel_values = mx.ones((1, image_size, image_size, num_channels))
    outputs = backbone(pixel_values)

    assert outputs.shape == (1, seq_len + 1, hidden_size)
