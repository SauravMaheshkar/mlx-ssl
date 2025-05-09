import argparse
from pathlib import Path

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download

from mlx_ssl.models.simclr.layers import get_resnet, name_to_params


parser = argparse.ArgumentParser(description="SimCLR converter for MLX")
parser.add_argument("model_id", type=str)
args = parser.parse_args()


def remap_key(key: str) -> str:
    # Skip keys that are not needed in MLX
    if key.endswith("num_batches_tracked"):
        return None

    # Stem remapping (PyTorch's first conv/bn to MLX's stem)
    if key.startswith("conv1."):
        key = key.replace("conv1.", "net.layers.0.layers.0.")
    if key.startswith("bn1."):
        key = key.replace("bn1.", "net.layers.0.layers.1.layers.0.")

    # FC layer remapping
    if key.startswith("fc."):
        key = key.replace("fc.", "fc.")  # If MLX uses a different name, change here

    # General remapping for blocks/layers
    parts = key.split(".")
    new_parts = []
    for part in parts:
        if part.isdigit():
            new_parts.append("layers")
            new_parts.append(part)
        else:
            new_parts.append(part)
    # Combine adjacent 'layers', e.g., layers.layers.0 -> layers.0
    result = []
    skip = False
    for i, part in enumerate(new_parts):
        if skip:
            skip = False
            continue
        if part == "layers" and i + 1 < len(new_parts) and new_parts[i + 1] == "layers":
            result.append("layers")
            skip = True
        else:
            result.append(part)
    return ".".join(result)


def port_to_mlx():
    path = Path(args.model_id)

    if not path.exists():
        path = Path(
            snapshot_download(
                repo_id="SauravMaheshkar/simclrv2-torch-weights",
                allow_patterns=[f"{args.model_id}.pth"],
            )
        )

    torch_weights = torch.load(path / f"{args.model_id}.pth")["resnet"]

    sanitized_weights = {}
    for k, v in torch_weights.items():
        remapped = remap_key(k)
        if remapped is not None:
            # Transpose conv weights from (O, I, H, W) to (O, H, W, I)
            if remapped.endswith(".weight") and v.ndim == 4:
                sanitized_weights[remapped] = mx.array(v.permute(0, 2, 3, 1))
            else:
                sanitized_weights[remapped] = mx.array(v)

    depth, width, sk_ratio = name_to_params(args.model_id)
    model, _ = get_resnet(depth, width, sk_ratio)

    model.load_weights(list(sanitized_weights.items()))
    model.save_weights(f"artifacts/{args.model_id}.safetensors")


if __name__ == "__main__":
    port_to_mlx()
