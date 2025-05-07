from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download


ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "silu": nn.silu,
}


def check_array_shape(arr: mx.array) -> bool:
    # PyTorch conv2d weight tensors have shape:
    #   [out_channels, in_channels, kH, KW]
    # MLX conv2d expects the weight be of shape:
    #   [out_channels, kH, KW, in_channels]
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    References:
        * https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/utils.py

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                ],
            )
        )
    return model_path
