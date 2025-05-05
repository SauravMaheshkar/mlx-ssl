import collections
import math
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

        self.projection = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, pixel_values: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        """

        Args:
            pixel_values: (batch_size, height, width, num_channels)

        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
            (patch_height, patch_width): the height and width of the patches
        """
        num_channels = pixel_values.shape[-1]

        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."  # noqa: E501
            )

        embeddings = self.projection(pixel_values)
        patch_height, patch_width = embeddings.shape[1], embeddings.shape[2]
        embeddings = embeddings.flatten(1, 2)

        return embeddings, (patch_height, patch_width)


class VisionEmbeddings(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        hidden_size: int,
        hidden_dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.cls_token = mx.zeros((1, 1, hidden_size))
        self.mask_token = None
        self.patch_encoder = PatchEncoder(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
        )
        self.patch_size = patch_size
        self.image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        self.position_embeddings = None
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def __call__(self, pixel_values: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        """

        Args:
            pixel_values: (batch_size, height, width, num_channels)

        Returns:
            embeddings: (batch_size, seq_len + 1, hidden_size)
            (patch_height, patch_width): the height and width of the patches
        """
        embeddings, (patch_height, patch_width) = self.patch_encoder(pixel_values)
        batch_size, _, _ = embeddings.shape

        cls_token = mx.repeat(self.cls_token, repeats=batch_size, axis=0)
        embeddings = mx.concatenate([cls_token, embeddings], axis=1)

        embeddings = self.dropout(embeddings)

        return embeddings, (patch_height, patch_width)


class VisionSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout_prob: float = 0.0,
    ):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention"  # noqa: E501
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout_prob)

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            context_layer: (batch_size, seq_len, hidden_size)
        """
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attn_scores = mx.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        attn_scores = attn_scores / math.sqrt(self.attention_head_size)

        attn_probs = nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs)

        context_layer = mx.matmul(attn_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer


class VisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        hidden_dropout_prob: float = 0.0,
        attention_dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.attn = VisionSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
        )
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        self_outputs = self.attn(hidden_states=hidden_states)
        attn_output = self.dropout(self.dense(self_outputs[0]))

        return attn_output
