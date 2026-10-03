from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from src.graph import combine_spatial_semantic, normalize_adjacency
from src.models.layers import DenseGraphConv, NodeBatchNorm


class CIFARResNet18Backbone(nn.Module):
    """Modified ResNet-18 producing a 512x4x4 feature map for CIFAR-10."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)

        net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        net.maxpool = nn.Identity()

        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CNNGCN(nn.Module):
    """ResNet18 + spatial/semantic graph + GCN classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        k: int = 4,
        dropout: float = 0.4,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.k = k
        self.grid_size = 4

        self.backbone = CIFARResNet18Backbone(pretrained_backbone)

        self.gcn1 = DenseGraphConv(512, 512)
        self.bn1 = NodeBatchNorm(512)
        self.gcn2 = DenseGraphConv(512, 256)
        self.bn2 = NodeBatchNorm(256)
        self.gcn3 = DenseGraphConv(256, 256)
        self.bn3 = NodeBatchNorm(256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def feature_map_to_nodes(fmap: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fmap.shape
        if (h, w) != (4, 4):
            raise RuntimeError(f"Expected 4x4 feature map, got {h}x{w}.")
        return fmap.flatten(2).transpose(1, 2).contiguous()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.feature_map_to_nodes(self.backbone(images))
        adj = combine_spatial_semantic(x, k=self.k, grid_size=self.grid_size)
        adj_norm = normalize_adjacency(adj)

        x = F.relu(self.bn1(self.gcn1(x, adj_norm)), inplace=True)
        x = F.relu(self.bn2(self.gcn2(x, adj_norm)), inplace=True)
        x = F.relu(self.bn3(self.gcn3(x, adj_norm)), inplace=True)

        graph_feature = torch.cat(
            [x.mean(dim=1), x.max(dim=1).values],
            dim=1,
        )
        return self.classifier(graph_feature)
