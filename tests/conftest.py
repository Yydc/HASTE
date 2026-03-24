import pytest
import torch
import numpy as np

from models import build_knn_gaussian


NUM_CHANNELS = 17
NUM_BANDS = 5
SEQ_LEN = 20  # small for fast tests


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dummy_adj():
    """Build a deterministic kNN adjacency from ring-layout electrodes."""
    angles = torch.linspace(0, 2 * torch.pi, NUM_CHANNELS + 1)[:-1]
    coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).numpy()
    return build_knn_gaussian(coords, k=4, sigma=0.5)


@pytest.fixture
def dummy_input(device):
    """Random input tensor: (batch=2, B=5, C=17, T=20)."""
    return torch.randn(2, NUM_BANDS, NUM_CHANNELS, SEQ_LEN, device=device)


@pytest.fixture
def dummy_target(device):
    """Random target tensor: (batch=2, T=20) in [0,1]."""
    return torch.rand(2, SEQ_LEN, device=device)


@pytest.fixture
def model_kwargs(dummy_adj):
    """Default model keyword arguments for testing."""
    return dict(
        num_channels=NUM_CHANNELS,
        num_bands=NUM_BANDS,
        spatial_d_model=32,   # smaller for speed
        spatial_heads=2,
        spatial_layers=1,
        temporal_d_model=64,
        temporal_heads=2,
        temporal_layers=1,
        temporal_window=5,
        temporal_ff_dim=128,
        d_r=16,
        d_embed=32,
        d_s=8,
        top_k=2,
        adjacency=dummy_adj,
        dropout=0.0,
    )
