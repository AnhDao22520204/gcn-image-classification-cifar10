from __future__ import annotations

import torch


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric GCN normalization with self-loops."""
    squeeze = False
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
        squeeze = True

    b, n, _ = adj.shape
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype).expand(b, n, n)
    a_tilde = torch.clamp(adj + eye, max=1.0)

    degree = a_tilde.sum(dim=-1).clamp_min(1e-8)
    d_inv_sqrt = degree.pow(-0.5)
    out = d_inv_sqrt.unsqueeze(-1) * a_tilde * d_inv_sqrt.unsqueeze(-2)
    return out.squeeze(0) if squeeze else out


def grid_coordinates(grid_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=dtype),
        torch.arange(grid_size, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def knn_adjacency_from_features(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Build an undirected k-NN adjacency matrix from [N,D] or [B,N,D]."""
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True

    b, n, _ = x.shape
    if n <= 1:
        out = torch.zeros((b, n, n), device=x.device, dtype=x.dtype)
        return out.squeeze(0) if squeeze else out

    k = max(1, min(k, n - 1))
    dist = torch.cdist(x.detach(), x.detach(), p=2)
    diagonal = torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(diagonal, float("inf"))

    idx = dist.topk(k=k, largest=False, dim=-1).indices
    adj = torch.zeros((b, n, n), device=x.device, dtype=x.dtype)
    adj.scatter_(2, idx, 1.0)
    adj = torch.maximum(adj, adj.transpose(1, 2))
    return adj.squeeze(0) if squeeze else adj


def grid_knn_adjacency(grid_size: int = 4, k: int = 4, device=None) -> torch.Tensor:
    coords = grid_coordinates(grid_size, device=device)
    return knn_adjacency_from_features(coords, k=k)


def four_neighbor_grid_adjacency(
    grid_size: int = 4,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    n = grid_size * grid_size
    adj = torch.zeros((n, n), device=device, dtype=dtype)

    def idx(r, c):
        return r * grid_size + c

    for r in range(grid_size):
        for c in range(grid_size):
            i = idx(r, c)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid_size and 0 <= cc < grid_size:
                    adj[i, idx(rr, cc)] = 1.0
    return torch.maximum(adj, adj.T)


def combine_spatial_semantic(
    node_features: torch.Tensor,
    k: int = 4,
    grid_size: int = 4,
) -> torch.Tensor:
    b, n, _ = node_features.shape
    expected = grid_size * grid_size
    if n != expected:
        raise ValueError(f"Expected {expected} nodes, got {n}.")

    spatial = four_neighbor_grid_adjacency(
        grid_size, node_features.device, node_features.dtype
    ).unsqueeze(0).expand(b, -1, -1)

    semantic = knn_adjacency_from_features(node_features, k=k)
    return torch.maximum(spatial, semantic)
