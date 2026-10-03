from __future__ import annotations

import torch
from torch import nn


class DenseGraphConv(nn.Module):
    """Dense GCN layer: H_next = A_norm @ H @ W."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if adj_norm.dim() == 2:
            return torch.matmul(adj_norm, h)
        return torch.bmm(adj_norm, h)


class NodeBatchNorm(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, f = x.shape
        x = self.bn(x.reshape(b * n, f))
        return x.reshape(b, n, f)
