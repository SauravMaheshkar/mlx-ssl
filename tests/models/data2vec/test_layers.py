import mlx.core as mx

from src.models.data2vec.layers import PatchEncoder


def test_patch_encoder():
    encoder = PatchEncoder(image_size=224, patch_size=16, num_channels=3, hidden_size=768)

    pixel_values = mx.ones((1, 224, 224, 3))
    embeddings, (patch_height, patch_width) = encoder(pixel_values)
    assert embeddings.shape == (1, 196, 768)
    assert patch_height == 14
    assert patch_width == 14
