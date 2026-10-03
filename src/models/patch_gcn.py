from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.graph import grid_knn_adjacency, normalize_adjacency
from src.models.layers import DenseGraphConv, NodeBatchNorm


class PatchGCN(nn.Module):
    """Patch-based GCN reconstructed from the academic report."""

    def __init__(
        self,
        num_classes: int = 10,
        patch_size: int = 8,
        embedding_dim: int = 128,
        k: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        if 32 % patch_size != 0:
            raise ValueError("patch_size must divide 32.")

        self.patch_size = patch_size
        self.grid_size = 32 // patch_size
        self.k = k

        patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, embedding_dim)

        self.gcn1 = DenseGraphConv(embedding_dim, 256)
        self.bn1 = NodeBatchNorm(256)
        self.gcn2 = DenseGraphConv(256, 256)
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

        adj = grid_knn_adjacency(self.grid_size, k=self.k)
        self.register_buffer("adj_norm", normalize_adjacency(adj))

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = F.unfold(images, kernel_size=p, stride=p)
        return patches.transpose(1, 2).contiguous()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(self.patchify(images))

        x = F.relu(self.bn1(self.gcn1(x, self.adj_norm)), inplace=True)
        x = F.relu(self.bn2(self.gcn2(x, self.adj_norm)), inplace=True)
        x = F.relu(self.bn3(self.gcn3(x, self.adj_norm)), inplace=True)

        graph_feature = torch.cat(
            [x.mean(dim=1), x.max(dim=1).values],
            dim=1,
        )
        return self.classifier(graph_feature)
