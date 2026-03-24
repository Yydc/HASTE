from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjacency import build_full_adjacency, row_normalize
from .blocks import (
    BandFunctionalPrior,
    CausalSpectralFusion,
    PriorMixingGate,
    SpatialTGALayer,
    TemporalBlock,
)


class SpatialEncoder(nn.Module):
    """
    Prior-biased spatial attention on the physical graph (Sec 4.3).
    Input: X_fused ∈ R^{batch×C×T}, prior P_t ∈ R^{batch×T×C×C}
    Output: H_spatial ∈ R^{batch×T×C×d_model}
    """

    def __init__(
        self,
        num_channels: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        adjacency: torch.Tensor,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model

        # Eq. 13: h^(0)_{i,t} = w_in * x_{i,t} + b_in
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_channels, d_model))

        self.layers = nn.ModuleList(
            [SpatialTGALayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        adj = adjacency.clone().detach()
        eye = torch.eye(num_channels, device=adj.device, dtype=adj.dtype)
        adj = torch.where(eye > 0, torch.ones_like(adj), adj)
        # Binarize for masking
        adj = (adj > 0).float()
        self.register_buffer("adjacency", adj)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.pos_embed)

    def forward(self, x_fused: torch.Tensor,
                prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_fused: (B, C, T)
        prior: (B, T, C, C) or None
        Returns: (B, T, C, d_model)
        """
        B, C, T = x_fused.shape
        h = x_fused.permute(0, 2, 1).unsqueeze(-1)  # (B, T, C, 1)
        h = self.input_proj(h)                      # (B, T, C, d_model)
        h = h + self.pos_embed

        adj = self.adjacency
        D = self.d_model

        for layer in self.layers:
            h_flat = h.reshape(B * T, C, D)
            # Get per-timestep prior if available
            if prior is not None:
                prior_flat = prior.reshape(B * T, C, C)
            else:
                prior_flat = None
            h_flat = layer(h_flat, adj, prior=prior_flat)
            h = h_flat.view(B, T, C, D)

        return h


class TemporalEncoder(nn.Module):
    """
    Tokenization + Causal Local Temporal Attention (Sec 4.3).
    Input H_spatial ∈ R^{batch×T×C×d_s}
    Tokenize via learned projection, then apply CLTA.
    """

    def __init__(
        self,
        num_channels: int,
        spatial_d_model: int,
        d_model: int,
        num_heads: int,
        window_size: int = 10,
        d_ff: int = 2048,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = num_channels * spatial_d_model
        self.d_model = d_model

        # Eq. 15: z_t = W_proj * vec(H_spatial_t) + b_proj
        self.token_proj = nn.Linear(self.in_dim, d_model)

        self.layers = nn.ModuleList(
            [
                TemporalBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    window_size=window_size,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, H_spatial: torch.Tensor) -> torch.Tensor:
        B, T, C, D = H_spatial.shape
        z = H_spatial.reshape(B, T, C * D)
        z = self.token_proj(z)  # (B, T, d_model)

        for layer in self.layers:
            z = layer(z)

        return z  # (B, T, d_model)


class RegressionHead(nn.Module):
    """
    Prediction head (Sec 4.3, Eq. 17).
    y_hat_t = sigma(w_out^T h_t + b_out)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, H_temporal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H_temporal: (B, T, d_model)
        Returns:
            y_final: (B,) prediction at last timestep
            y_seq: (B, T) predictions at all timesteps
        """
        y_seq = torch.sigmoid(self.fc(H_temporal)).squeeze(-1)  # (B, T)
        y_final = y_seq[:, -1]  # (B,)
        return y_final, y_seq


class HASTModel(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_bands: int,
        spatial_d_model: int = 64,
        spatial_heads: int = 4,
        spatial_layers: int = 2,
        temporal_d_model: int = 128,
        temporal_heads: int = 4,
        temporal_layers: int = 1,
        temporal_window: int = 10,
        temporal_ff_dim: int = 512,
        d_r: int = 32,
        d_embed: int = 64,
        d_s: int = 16,
        top_k: int = 3,
        adjacency: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.top_k = top_k

        if adjacency is None:
            adjacency = build_full_adjacency(num_channels)

        # Binary adjacency for masking
        adj_binary = (adjacency > 0).float()
        self.register_buffer("adj_binary", adj_binary)

        # Row-normalized adjacency as anatomical prior P^phy (Eq. 10)
        P_phy = row_normalize(adj_binary)
        self.register_buffer("P_phy", P_phy)

        # 1. Causal Spectral Fusion (GRU-based)
        self.spectral_fusion = CausalSpectralFusion(
            num_bands=num_bands,
            d_r=d_r,
        )

        # 2. Band-wise Functional Prior
        self.func_prior = BandFunctionalPrior(
            num_bands=num_bands,
            d_embed=d_embed,
            d_s=d_s,
        )

        # 3. Prior-Mixing Gate
        self.prior_gate = PriorMixingGate(d_embed=d_embed)

        # 4. Spatial Encoder (with log-bias attention)
        self.spatial_encoder = SpatialEncoder(
            num_channels=num_channels,
            d_model=spatial_d_model,
            num_heads=spatial_heads,
            num_layers=spatial_layers,
            adjacency=adjacency,
            dropout=dropout,
        )

        # 5. Temporal Encoder (with learned tokenization)
        self.temporal_encoder = TemporalEncoder(
            num_channels=num_channels,
            spatial_d_model=spatial_d_model,
            d_model=temporal_d_model,
            num_heads=temporal_heads,
            window_size=temporal_window,
            d_ff=temporal_ff_dim,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # 6. Regression Head
        self.regression_head = RegressionHead(d_model=temporal_d_model)

    def _topk_mask_and_weights(
        self, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TopK band gating (Sec 4.2, Eq. 11).
        alpha: (batch, T, B)
        Returns:
            mask: (batch, T, B) binary
            renorm_alpha: (batch, T, B) renormalized weights
        """
        B_bands = alpha.shape[-1]
        K = min(self.top_k, B_bands)

        _, topk_idx = alpha.topk(K, dim=-1)  # (batch, T, K)
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, topk_idx, 1.0)

        masked_alpha = alpha * mask
        renorm_alpha = masked_alpha / masked_alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return mask, renorm_alpha

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        x: (batch, B, C, T)
        Returns:
            y_final: (batch,)
            y_seq: (batch, T)
            alpha: (batch, T, B) band weights
            P_func: (batch, T, C, C) functional prior
            g: (batch, T, 1, 1) prior-mixing gate
            P_t: (batch, T, C, C) fused prior
        """
        batch, B, C, T = x.shape

        # 1. Causal spectral fusion
        X_fused, alpha = self.spectral_fusion(x)  # (batch, C, T), (batch, T, B)

        # 2. TopK band gating
        active_mask, active_alpha = self._topk_mask_and_weights(alpha)

        # 3. Band-wise functional prior
        P_func, embed_mean = self.func_prior(
            x, self.adj_binary, active_mask, active_alpha
        )  # (batch, T, C, C), (batch, T, d_embed)

        # 4. Prior-mixing gate
        g = self.prior_gate(embed_mean)  # (batch, T, 1, 1)

        # 5. Fuse priors: P_t = g * P^phy + (1 - g) * P^func_t
        P_phy_exp = self.P_phy.unsqueeze(0).unsqueeze(0)  # (1, 1, C, C)
        P_t = g * P_phy_exp + (1 - g) * P_func  # (batch, T, C, C)

        # 6. Spatial encoding with prior
        H_spatial = self.spatial_encoder(X_fused, prior=P_t)

        # 7. Temporal encoding
        H_temporal = self.temporal_encoder(H_spatial)

        # 8. Regression
        y_final, y_seq = self.regression_head(H_temporal)

        return y_final, y_seq, alpha, P_func, g, P_t


def hast_loss_fn(
    y_true: torch.Tensor,
    y_final: torch.Tensor,
    y_seq: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    P_func: Optional[torch.Tensor] = None,
    g: Optional[torch.Tensor] = None,
    lambda_pred: float = 0.1,
    lambda_band: float = 0.01,
    lambda_graph: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full HASTE loss (Sec 4.4, Eq. 19-22).

    L = L_reg + lambda_pred * L_pred + lambda_band * L_band + lambda_graph * L_graph

    L_reg   = (1/T) sum_t (y_t - y_hat_t)^2
    L_pred  = (1/(T-1)) sum_t (y_hat_t - y_hat_{t-1})^2
    L_band  = (1/(T-1)) sum_t ||alpha_t - alpha_{t-1}||^2
    L_graph = (1/(T-1)) sum_t ||P^func_t - P^func_{t-1}||_F^2
            + (1/(T-1)) sum_t (g_t - g_{t-1})^2
    """
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)

    # L_reg: MSE over all timesteps (Eq. 20)
    L_reg = F.mse_loss(y_seq, y_true)

    # L_pred: prediction smoothness (Eq. 21)
    T = y_seq.shape[1]
    if T > 1:
        L_pred = ((y_seq[:, 1:] - y_seq[:, :-1]) ** 2).mean()
    else:
        L_pred = torch.tensor(0.0, device=y_seq.device)

    # L_band: band drift regularization (Eq. 9)
    if alpha is not None and alpha.shape[1] > 1:
        L_band = ((alpha[:, 1:, :] - alpha[:, :-1, :]) ** 2).mean()
    else:
        L_band = torch.tensor(0.0, device=y_seq.device)

    # L_graph: functional prior drift + gate drift (Eq. 22)
    L_graph = torch.tensor(0.0, device=y_seq.device)
    if P_func is not None and P_func.shape[1] > 1:
        L_graph = L_graph + ((P_func[:, 1:] - P_func[:, :-1]) ** 2).mean()
    if g is not None and g.shape[1] > 1:
        g_flat = g.squeeze(-1).squeeze(-1)  # (batch, T)
        L_graph = L_graph + ((g_flat[:, 1:] - g_flat[:, :-1]) ** 2).mean()

    total_loss = L_reg + lambda_pred * L_pred + lambda_band * L_band + lambda_graph * L_graph

    return total_loss, L_reg, L_pred, L_band, L_graph


__all__ = [
    "HASTModel",
    "SpatialEncoder",
    "TemporalEncoder",
    "RegressionHead",
    "build_full_adjacency",
    "hast_loss_fn",
]
