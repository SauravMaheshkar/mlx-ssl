import mlx.core as mx
import pytest

from mlx_ssl.models.data2vec.layers import (
    Data2VecVisionEmbeddings,
    PatchEncoder,
    VisionAttention,
    VisionEncoder,
    VisionLayer,
    VisionSelfAttention,
)


@pytest.mark.parametrize(
    "path_or_hf_repo,revision",
    [
        ("mlx-community/data2vec-vision-large-ft1k", "main"),
    ],
)
def test_patch_encoder(fetch_config):
    encoder = PatchEncoder(config=fetch_config)

    pixel_values = mx.ones(
        (1, fetch_config.image_size, fetch_config.image_size, fetch_config.num_channels)
    )
    embeddings, (height, width) = encoder(pixel_values)
    assert embeddings.shape == (1, 196, fetch_config.hidden_size)
    assert height == 14
    assert width == 14


@pytest.mark.parametrize(
    "path_or_hf_repo,revision",
    [
        ("mlx-community/data2vec-vision-large-ft1k", "main"),
    ],
)
def test_vision_embeddings(fetch_config):
    module = Data2VecVisionEmbeddings(config=fetch_config)

    pixel_values = mx.ones(
        (1, fetch_config.image_size, fetch_config.image_size, fetch_config.num_channels)
    )
    embeddings, (height, width) = module(pixel_values)
    assert embeddings.shape == (1, 196 + 1, fetch_config.hidden_size)
    assert height == 14
    assert width == 14


@pytest.mark.parametrize(
    "path_or_hf_repo,revision",
    [
        ("mlx-community/data2vec-vision-large-ft1k", "main"),
    ],
)
def test_vision_self_attention(fetch_config):
    attn_module = VisionSelfAttention(config=fetch_config)
    encoder = PatchEncoder(config=fetch_config)

    pixel_values = mx.ones(
        (1, fetch_config.image_size, fetch_config.image_size, fetch_config.num_channels)
    )
    hidden_states, _ = encoder(pixel_values)

    context_layer = attn_module(hidden_states)
    assert context_layer.shape == (1, 196, fetch_config.hidden_size)


@pytest.mark.parametrize(
    "path_or_hf_repo,revision",
    [
        ("mlx-community/data2vec-vision-large-ft1k", "main"),
    ],
)
def test_vision_attention(fetch_config):
    attn = VisionAttention(config=fetch_config)
    encoder = PatchEncoder(config=fetch_config)

    pixel_values = mx.ones(
        (1, fetch_config.image_size, fetch_config.image_size, fetch_config.num_channels)
    )
    hidden_states, _ = encoder(pixel_values)
    attn_out = attn(hidden_states)

    assert attn_out.shape == (196, fetch_config.hidden_size)


@pytest.mark.parametrize(
    "path_or_hf_repo,revision",
    [
        ("mlx-community/data2vec-vision-large-ft1k", "main"),
    ],
)
def test_vision_layer(fetch_config):
    layer = VisionLayer(config=fetch_config)
    encoder = PatchEncoder(config=fetch_config)

    pixel_values = mx.ones(
        (1, fetch_config.image_size, fetch_config.image_size, fetch_config.num_channels)
    )
    hidden_states, _ = encoder(pixel_values)
    layer_out = layer(hidden_states)

    assert layer_out.shape == (1, 196, fetch_config.hidden_size)


@pytest.mark.parametrize(
    "path_or_hf_repo,revision",
    [
        ("mlx-community/data2vec-vision-large-ft1k", "main"),
    ],
)
def test_vision_encoder(fetch_config):
    encoder = VisionEncoder(config=fetch_config)

    encoder_module = PatchEncoder(config=fetch_config)

    pixel_values = mx.ones(
        (1, fetch_config.image_size, fetch_config.image_size, fetch_config.num_channels)
    )
    hidden_states, _ = encoder_module(pixel_values)
    encoder_out = encoder(hidden_states)

    assert encoder_out.shape == (1, 196, fetch_config.hidden_size)
