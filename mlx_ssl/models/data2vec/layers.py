import collections
import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
from ml_collections.config_dict import ConfigDict

from ...utils import ACT2FN


class DropPath(nn.Module):
    def __init__(self, config: ConfigDict) -> None:
        self.drop_path_rate = config.drop_path_rate

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.drop_path_rate == 0.0:
            return hidden_states

        keep_prob = 1 - self.drop_path_rate
        shape = (hidden_states.shape[0],) + (1,) * (len(hidden_states.shape) - 1)
        random_tensor = mx.random.uniform(
            shape=shape, low=0.0, high=1.0, dtype=hidden_states.dtype
        )
        random_tensor = mx.floor(random_tensor)
        output = mx.divide(hidden_states, keep_prob) * random_tensor
        return output


class PatchEncoder(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ) -> None:
        super().__init__()

        image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = config.num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
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


class Data2VecVisionEmbeddings(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        super().__init__()

        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        self.mask_token = None
        self.patch_embeddings = PatchEncoder(config)
        self.patch_size = config.patch_size
        self.image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, pixel_values: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        """

        Args:
            pixel_values: (batch_size, height, width, num_channels)

        Returns:
            embeddings: (batch_size, seq_len + 1, hidden_size)
            (patch_height, patch_width): the height and width of the patches
        """
        embeddings, (patch_height, patch_width) = self.patch_embeddings(pixel_values)
        batch_size, _, _ = embeddings.shape

        cls_token = mx.repeat(self.cls_token, repeats=batch_size, axis=0)
        embeddings = mx.concatenate([cls_token, embeddings], axis=1)

        embeddings = self.dropout(embeddings)

        return embeddings, (patch_height, patch_width)


class VisionSelfAttention(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention"  # noqa: E501
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
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


class VisionAttentionOutput(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class VisionAttention(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        self.attention = VisionSelfAttention(
            config=config,
        )
        self.output = VisionAttentionOutput(config)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        self_outputs = self.attention(hidden_states=hidden_states)
        attn_output = self.output(self_outputs[0])

        return attn_output


class Data2VecVisionIntermediate(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Data2VecVisionOutput(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class VisionLayer(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
        drop_path_rate: float = 0.0,
    ):
        self.seq_len_dim = 1
        self.attention = VisionAttention(config)

        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.drop_path = DropPath(ConfigDict({"drop_path_rate": drop_path_rate}))
        self.intermediate = Data2VecVisionIntermediate(config)
        self.output = Data2VecVisionOutput(config)

        if config.layer_scale_init_value > 0.0:
            self.lambda_1 = config.layer_scale_init_value * mx.ones((config.hidden_size,))
            self.lambda_2 = config.layer_scale_init_value * mx.ones((config.hidden_size,))
        else:
            self.lambda_1 = None
            self.lambda_2 = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            layer_output: (batch_size, seq_len, hidden_size)
        """
        self_attn_outputs = self.attention(
            hidden_states=self.layernorm_before(hidden_states)
        )

        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * self_attn_outputs

        hidden_states = self.drop_path(attention_output) + hidden_states
        layer_output = self.layernorm_after(hidden_states)

        layer_out = self.intermediate(layer_output)
        layer_out = self.output(layer_out)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_out

        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output + self_attn_outputs


class VisionEncoder(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        dpr = [
            x.item()
            for x in mx.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]
        self.layer = nn.Sequential(
            *[
                VisionLayer(config, drop_path_rate=dpr[i])
                for i in range(config.num_hidden_layers)
            ]
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.layer(hidden_states)
        return hidden_states


class Data2VecVisionPooler(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ) -> None:
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.use_mean_pooling
            else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = mx.mean(self.layernorm(patch_tokens), axis=1)
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output
