from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet9(nn.Module):
    """Small OpenGait-style ResNet9 frame backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | tuple[int, ...] = (64, 128, 256, 512),
        layers: list[int] | tuple[int, ...] = (1, 1, 1, 1),
        strides: list[int] | tuple[int, ...] = (1, 2, 2, 1),
        maxpool: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = int(channels[0])
        self.maxpool_enabled = bool(maxpool)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = BasicConv2d(in_channels, self.inplanes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(int(channels[0]), int(layers[0]), int(strides[0]))
        self.layer2 = self._make_layer(int(channels[1]), int(layers[1]), int(strides[1]))
        self.layer3 = self._make_layer(int(channels[2]), int(layers[2]), int(strides[2]))
        self.layer4 = self._make_layer(int(channels[3]), int(layers[3]), int(strides[3]))

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Module:
        if blocks <= 0:
            return nn.Identity()
        layers = [BasicBlock(self.inplanes, planes, stride=stride)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        if self.maxpool_enabled:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SetBlockWrapper(nn.Module):
    """Apply a 2D frame model independently over a gait sequence."""

    def __init__(self, frame_model: nn.Module) -> None:
        super().__init__()
        self.frame_model = frame_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        x = x.transpose(1, 2).reshape(batch * frames, channels, height, width)
        x = self.frame_model(x)
        _, out_channels, out_height, out_width = x.shape
        x = x.reshape(batch, frames, out_channels, out_height, out_width)
        return x.transpose(1, 2).contiguous()


class HorizontalPoolingPyramid(nn.Module):
    def __init__(self, bin_num: list[int] | tuple[int, ...] = (16,)) -> None:
        super().__init__()
        self.bin_num = tuple(int(bin_size) for bin_size in bin_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, _ = x.shape
        features = []
        for bin_size in self.bin_num:
            if height % bin_size != 0:
                raise ValueError(f"Feature height {height} is not divisible by HPP bin {bin_size}.")
            pooled = x.view(batch, channels, bin_size, -1)
            pooled = pooled.mean(-1) + pooled.max(-1)[0]
            features.append(pooled)
        return torch.cat(features, dim=-1)


class SeparateFCs(nn.Module):
    def __init__(self, parts_num: int, in_channels: int, out_channels: int, norm: bool = False) -> None:
        super().__init__()
        self.norm = bool(norm)
        self.fc_bin = nn.Parameter(torch.empty(parts_num, in_channels, out_channels))
        nn.init.xavier_uniform_(self.fc_bin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1).contiguous()
        weight = F.normalize(self.fc_bin, dim=1) if self.norm else self.fc_bin
        x = x.matmul(weight)
        return x.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    def __init__(self, parts_num: int, in_channels: int, class_num: int, norm: bool = True) -> None:
        super().__init__()
        self.norm = bool(norm)
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        self.fc_bin = nn.Parameter(torch.empty(parts_num, in_channels, class_num))
        nn.init.xavier_uniform_(self.fc_bin)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, parts = x.shape
        feature = self.bn1d(x.view(batch, -1)).view(batch, channels, parts)
        feature = feature.permute(2, 0, 1).contiguous()
        weight = F.normalize(self.fc_bin, dim=1) if self.norm else self.fc_bin
        if self.norm:
            feature = F.normalize(feature, dim=-1)
        logits = feature.matmul(weight)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class ProjectionGaitBaseline(nn.Module):
    """Projection baseline aligned with OpenGait's ResNet9 + HPP recipe.

    This model is intentionally small and self-contained.  It gives us a stable
    baseline that can later be swapped with GaitBase, DeepGaitV2, LidarGait++,
    GaitCloud, or any other method through the same training/evaluation code.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        channels: list[int] | tuple[int, ...] = (64, 128, 256, 512),
        layers: list[int] | tuple[int, ...] = (1, 1, 1, 1),
        strides: list[int] | tuple[int, ...] = (1, 2, 2, 1),
        maxpool: bool = False,
        embedding_dim: int = 256,
        parts_num: int = 16,
        bin_num: list[int] | tuple[int, ...] = (16,),
    ) -> None:
        super().__init__()
        self.backbone = SetBlockWrapper(
            ResNet9(
                in_channels=in_channels,
                channels=channels,
                layers=layers,
                strides=strides,
                maxpool=maxpool,
            )
        )
        self.temporal_pool = torch.max
        self.hpp = HorizontalPoolingPyramid(bin_num=bin_num)
        self.fcs = SeparateFCs(parts_num=parts_num, in_channels=int(channels[-1]), out_channels=embedding_dim)
        self.bn_necks = SeparateBNNecks(parts_num=parts_num, in_channels=embedding_dim, class_num=num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.affine:
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("Expected images with shape [B, T, C, H, W].")
        x = images.permute(0, 2, 1, 3, 4).contiguous()
        x = self.backbone(x)
        x = self.temporal_pool(x, dim=2)[0]
        pooled = self.hpp(x)
        embeddings = self.fcs(pooled)
        bn_embeddings, part_logits = self.bn_necks(embeddings)
        flat_embedding = bn_embeddings.flatten(1)
        return {
            "embeddings": embeddings,
            "bn_embeddings": bn_embeddings,
            "embedding": flat_embedding,
            "part_logits": part_logits,
            "logits": part_logits.mean(dim=2),
        }


def part_cross_entropy(part_logits: torch.Tensor, labels: torch.Tensor, scale: float = 16.0) -> torch.Tensor:
    losses = []
    scaled = part_logits * float(scale)
    for part_idx in range(scaled.size(2)):
        losses.append(F.cross_entropy(scaled[:, :, part_idx], labels))
    return torch.stack(losses).mean()


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("name", "projection_gait_baseline")
    if name != "projection_gait_baseline":
        raise ValueError(f"Unknown model name: {name}")
    return ProjectionGaitBaseline(
        num_classes=int(cfg.get("num_classes", 29)),
        in_channels=int(cfg.get("in_channels", 3)),
        channels=cfg.get("channels", [64, 128, 256, 512]),
        layers=cfg.get("layers", [1, 1, 1, 1]),
        strides=cfg.get("strides", [1, 2, 2, 1]),
        maxpool=bool(cfg.get("maxpool", False)),
        embedding_dim=int(cfg.get("embedding_dim", 256)),
        parts_num=int(cfg.get("parts_num", 16)),
        bin_num=cfg.get("bin_num", [16]),
    )
