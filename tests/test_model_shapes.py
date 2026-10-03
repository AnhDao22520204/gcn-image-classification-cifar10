import torch

from src.graph import grid_knn_adjacency, normalize_adjacency
from src.models import PatchGCN, CNNGCN


def test_patch_gcn_output_shape():
    model = PatchGCN()
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 10)


def test_cnn_gcn_output_shape():
    model = CNNGCN()
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 10)


def test_grid_graph_shape():
    adj = grid_knn_adjacency(grid_size=4, k=4)
    assert adj.shape == (16, 16)
    assert normalize_adjacency(adj).shape == (16, 16)
