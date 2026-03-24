"""Tests for the HASTE model architecture."""

import torch
import pytest

from models import HASTModel, hast_loss_fn, build_full_adjacency, build_knn_gaussian


def test_forward_shapes(model_kwargs, dummy_input, dummy_target, device):
    """Model output shapes match expectations."""
    model = HASTModel(**model_kwargs).to(device)
    model.eval()

    B, Bands, C, T = dummy_input.shape

    with torch.no_grad():
        y_final, y_seq, alpha, P_func, g, P_t = model(dummy_input)

    assert y_final.shape == (B,)
    assert y_seq.shape == (B, T)
    assert alpha.shape == (B, T, Bands)
    assert P_func.shape == (B, T, C, C)
    assert g.shape == (B, T, 1, 1)
    assert P_t.shape == (B, T, C, C)


def test_output_range(model_kwargs, dummy_input, device):
    """Predictions should be in [0, 1] due to sigmoid."""
    model = HASTModel(**model_kwargs).to(device)
    model.eval()

    with torch.no_grad():
        y_final, y_seq, _, _, _, _ = model(dummy_input)

    assert (y_final >= 0).all() and (y_final <= 1).all()
    assert (y_seq >= 0).all() and (y_seq <= 1).all()


def test_backward_pass(model_kwargs, dummy_input, dummy_target, device):
    """Gradients flow through the entire model."""
    model = HASTModel(**model_kwargs).to(device)
    model.train()

    y_final, y_seq, alpha, P_func, g, P_t = model(dummy_input)
    loss, _, _, _, _ = hast_loss_fn(
        y_true=dummy_target, y_final=y_final, y_seq=y_seq,
        alpha=alpha, P_func=P_func, g=g,
    )
    loss.backward()

    n_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "No parameters received gradients"


def test_loss_finite(model_kwargs, dummy_input, dummy_target, device):
    """Loss values should be finite and non-negative."""
    model = HASTModel(**model_kwargs).to(device)
    model.eval()

    with torch.no_grad():
        y_final, y_seq, alpha, P_func, g, P_t = model(dummy_input)
        total, reg, pred, band, graph = hast_loss_fn(
            y_true=dummy_target, y_final=y_final, y_seq=y_seq,
            alpha=alpha, P_func=P_func, g=g,
        )

    for name, val in [("total", total), ("reg", reg), ("pred", pred),
                       ("band", band), ("graph", graph)]:
        assert torch.isfinite(val), f"{name} loss is not finite"
        assert val >= 0, f"{name} loss is negative"


def test_topk_gating(model_kwargs, dummy_input, device):
    """TopK mask should have exactly K active bands per timestep."""
    model = HASTModel(**model_kwargs).to(device)
    model.eval()
    K = model_kwargs["top_k"]

    with torch.no_grad():
        y_final, y_seq, alpha, _, _, _ = model(dummy_input)

    # Manually check topk
    mask, _ = model._topk_mask_and_weights(alpha)
    active_count = mask.sum(dim=-1)  # (B, T)
    assert (active_count == K).all(), f"Expected {K} active bands, got varying counts"


def test_gate_range(model_kwargs, dummy_input, device):
    """Prior-mixing gate g should be in [0, 1]."""
    model = HASTModel(**model_kwargs).to(device)
    model.eval()

    with torch.no_grad():
        _, _, _, _, g, _ = model(dummy_input)

    assert (g >= 0).all() and (g <= 1).all()


def test_alpha_sum_to_one(model_kwargs, dummy_input, device):
    """Band weights alpha should sum to 1 across bands."""
    model = HASTModel(**model_kwargs).to(device)
    model.eval()

    with torch.no_grad():
        _, _, alpha, _, _, _ = model(dummy_input)

    sums = alpha.sum(dim=-1)  # (B, T)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_knn_adjacency_properties():
    """kNN adjacency should be symmetric with self-loops."""
    import numpy as np
    coords = np.random.randn(17, 2).astype(np.float32)
    adj = build_knn_gaussian(coords, k=4, sigma=1.0, self_loop=True)

    # Symmetric
    assert torch.allclose(adj, adj.T, atol=1e-6)

    # Self-loops
    diag = torch.diag(adj)
    assert (diag > 0).all()

    # Non-negative
    assert (adj >= 0).all()


def test_full_adjacency():
    """Full adjacency should be all ones."""
    adj = build_full_adjacency(10)
    assert adj.shape == (10, 10)
    assert (adj == 1).all()
