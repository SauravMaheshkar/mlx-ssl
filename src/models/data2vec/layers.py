import collections
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class PatchEncoder(nn.Module):
    def __init__(
        self, image_size: int, patch_size: int, num_channels: int, hidden_size: int
    ):
        super().__init__()

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        print(
            f"in_channels: {num_channels}, out_channels: {hidden_size}, kernel_size: {patch_size}, stride: {patch_size}"  # noqa: E501
        )

        self.projection = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, pixel_values: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        num_channels = pixel_values.shape[-1]

        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."  # noqa: E501
            )

        embeddings = self.projection(pixel_values)
        patch_height, patch_width = embeddings.shape[1], embeddings.shape[2]
        embeddings = embeddings.flatten(1, 2)

        return embeddings, (patch_height, patch_width)
