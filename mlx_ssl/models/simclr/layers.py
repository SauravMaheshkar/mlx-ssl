from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Optional[int] = 3,
    stride: Optional[int] = 1,
    bias: Optional[bool] = False,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=bias,
    )


class BatchNormRelu(nn.Sequential):
    def __init__(self, num_channels, relu=True):
        super().__init__(
            nn.BatchNorm(num_channels, eps=BATCH_NORM_EPSILON),
            nn.ReLU() if relu else nn.Identity(),
        )


class ZeroPad2d:
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        self.padding = padding

        # Convert padding to standard format for mlx.core.pad
        if isinstance(padding, int):
            # Same padding on all sides
            self.pad_width = [(0, 0), (0, 0), (padding, padding), (padding, padding)]
        else:
            # padding is a 4-tuple (left, right, top, bottom)
            left, right, top, bottom = padding
            self.pad_width = [(0, 0), (0, 0), (top, bottom), (left, right)]

    def __call__(self, x):
        return mx.pad(x, pad_width=self.pad_width, mode="constant", constant_values=0)


class SelectiveKernel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        sk_ratio: float,
        min_dim: Optional[int] = 32,
    ) -> None:
        assert sk_ratio > 0.0

        self.main_conv = nn.Sequential(
            conv(in_channels, 2 * out_channels, stride=stride),
            BatchNormRelu(2 * out_channels),
        )

        mid_dim = max(int(out_channels * sk_ratio), min_dim)

        self.mixing_conv = nn.Sequential(
            conv(out_channels, mid_dim, kernel_size=1),
            BatchNormRelu(mid_dim),
            conv(mid_dim, 2 * out_channels, kernel_size=1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.main_conv(x)
        x = mx.stack(mx.split(x, 2, axis=1), axis=0)
        g = mx.sum(x, axis=0).mean(axis=[1, 2], keepdims=True)
        m = self.mixing_conv(g)
        m = mx.stack(mx.split(m, 2, axis=1), axis=0)

        return (x * mx.softmax(m, axis=0)).sum(axis=0)


class Projection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        sk_ratio: Optional[float] = 0.0,
    ) -> None:
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(
                ZeroPad2d((0, 1, 0, 1)),
                nn.AvgPool2d(kernel_size=2, stride=stride, padding=0),
                conv(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)

        self.bn = BatchNormRelu(out_channels, relu=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.shortcut(x))


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False
    ):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [
            conv(in_channels, out_channels, kernel_size=1),
            BatchNormRelu(out_channels),
        ]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(BatchNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(BatchNormRelu(out_channels * 4, relu=False))
        self.net = nn.Sequential(*ops)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = self.projection(x)
        return nn.relu(shortcut + self.net(x))


class Blocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = [Bottleneck(in_channels, out_channels, stride, sk_ratio, True)]
        self.channels_out = out_channels * Bottleneck.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(Bottleneck(self.channels_out, out_channels, 1, sk_ratio))

        self.blocks = nn.Sequential(*self.blocks)

    def __call__(self, x: mx.array) -> mx.array:
        return self.blocks(x)


class Stem(nn.Sequential):
    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(BatchNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)


class ResNet(nn.Module):
    def __init__(self, layers, width_multiplier, sk_ratio):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)
        self.fc = nn.Linear(channels_in, 1000)

    def __call__(self, x: mx.array, apply_fc: bool = False) -> mx.array:
        h = self.net(x).mean(axis=[1, 2])
        if apply_fc:
            h = self.fc(h)
        return h


class ContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                bn.bias = mx.array(0)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


def get_resnet(depth=50, width_multiplier=1, sk_ratio=0):
    layers = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }[depth]
    resnet = ResNet(layers, width_multiplier, sk_ratio)
    return resnet, ContrastiveHead(resnet.channels_out)


def name_to_params(checkpoint):
    sk_ratio = 0.0625 if "_sk1" in checkpoint else 0
    if "r50_" in checkpoint:
        depth = 50
    elif "r101_" in checkpoint:
        depth = 101
    elif "r152_" in checkpoint:
        depth = 152
    else:
        raise NotImplementedError

    if "_1x_" in checkpoint:
        width = 1
    elif "_2x_" in checkpoint:
        width = 2
    elif "_3x_" in checkpoint:
        width = 3
    else:
        raise NotImplementedError

    return depth, width, sk_ratio
