import json

import mlx.core as mx
import mlx.nn as nn
from ml_collections.config_dict import ConfigDict

from ...utils import check_array_shape, get_model_path
from .layers import (
    Data2VecVisionEmbeddings,
    Data2VecVisionPooler,
    VisionEncoder,
)


class Data2VecVisionForImageClassification(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        self.config = config
        self.embeddings = Data2VecVisionEmbeddings(config)
        self.encoder = VisionEncoder(config)

        self.layernorm = (
            nn.Identity()
            if config.use_mean_pooling
            else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pooler = Data2VecVisionPooler(config) if config.use_mean_pooling else None

        self.classifier = (
            nn.Linear(
                input_dims=config.hidden_size, output_dims=config.num_labels, bias=True
            )
            if config.num_labels > 0
            else nn.Identity()
        )

    @classmethod
    def from_pretrained(cls, path_or_hf_repo: str):
        path = get_model_path(path_or_hf_repo)

        ## load and sanitize weights
        weights = mx.load((path / "model.safetensors").as_posix())
        sanitized_weights = cls.sanitize(weights)

        ## load and sanitize config
        hf_config = json.loads(open(path / "config.json").read())
        hf_config["num_labels"] = len(hf_config["id2label"])
        config = ConfigDict(hf_config, allow_dotted_keys=True)

        ## initialize model
        model = cls(config)

        model.load_weights(list(sanitized_weights.items()))

        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}

        for k, v in weights.items():
            k = k.replace("data2vec_vision.", "")

            if "relative_position_bias" in k:
                continue

            if "encoder.layer." in k:
                k = k.replace("encoder.layer.", "encoder.layer.layers.")

            if "embeddings.patch_embeddings.projection.weight" in k:
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights

    def __call__(self, pixel_values: mx.array) -> mx.array:
        ## compute embeddings
        embedding_output, _ = self.embeddings(pixel_values)

        ## compute encoder output
        encoder_output = self.encoder(embedding_output)

        ## compute sequence output
        sequence_output = self.layernorm(encoder_output)

        ## compute pooled output
        pooled_output = self.pooler(sequence_output)

        ## compute logits
        logits = self.classifier(pooled_output)

        return logits
