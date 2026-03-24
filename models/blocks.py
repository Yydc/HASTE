import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSpectralFusion(nn.Module):
    """
    GRU-based causal spectral fusion (Sec 4.2).
    At each time step t, computes band relevance alpha_t via GRU,
    and fuses multi-band features into a single scalar per electrode.

    Input:  X ∈ R^{B, C, T, B_bands}  -- but we receive (batch, B_bands, C, T)
    Output: X_fused ∈ R^{batch, C, T}, alpha ∈ R^{batch, T, B_bands}
    """

    def __init__(self, num_bands: int, d_r: int = 32):
        super().__init__()
        self.num_bands = num_bands
        self.d_r = d_r

        # GRU: input is band descriptor u_t ∈ R^B, hidden state r_t ∈ R^d_r
        self.gru = nn.GRUCell(num_bands, d_r)
        # Project GRU output to band weights
        self.fc = nn.Linear(d_r, num_bands)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, B, C, T)
        Returns:
            X_fused: (batch, C, T)
            alpha: (batch, T, B) -- band weights at each timestep
        """
        batch_size, B, C, T = x.shape
        assert B == self.num_bands

        # Compute global band descriptor per timestep: u_t = mean over electrodes
        # x: (batch, B, C, T) -> (batch, T, B) by averaging over C
        u = x.mean(dim=2).permute(0, 2, 1)  # (batch, T, B)

        # Run GRU causally over time
        r = torch.zeros(batch_size, self.d_r, device=x.device, dtype=x.dtype)
        alphas = []
        for t in range(T):
            r = self.gru(u[:, t, :], r)  # (batch, d_r)
            logits = self.fc(r)           # (batch, B)
            alpha_t = F.softmax(logits, dim=-1)  # (batch, B)
            alphas.append(alpha_t)

        alpha = torch.stack(alphas, dim=1)  # (batch, T, B)

        # Fuse: x_{c,t} = sum_b alpha_{t,b} * X_{c,t,b}
        # x: (batch, B, C, T), alpha: (batch, T, B)
        # Rearrange for broadcasting
        alpha_exp = alpha.permute(0, 2, 1).unsqueeze(2)  # (batch, B, 1, T)
        X_fused = (alpha_exp * x).sum(dim=1)  # (batch, C, T)

        return X_fused, alpha


class BandFunctionalPrior(nn.Module):
    """
    Computes band-wise functional priors P^func_{t,b} on the physical graph (Sec 4.2, Eqs. 13-15).
    For each active band, embeds electrode features, computes low-dim similarity,
    and produces a row-stochastic prior via neighborhood softmax.
    """

    def __init__(self, num_bands: int, d_embed: int = 64, d_s: int = 16):
        super().__init__()
        self.num_bands = num_bands
        self.d_embed = d_embed
        self.d_s = d_s

        # Per-band embedding: e_{c,t,b} = W_b * X_{c,t,b} + b_b
        self.band_embeds = nn.ModuleList([
            nn.Linear(1, d_embed) for _ in range(num_bands)
        ])
        # Shared low-dim projections for similarity
        self.W_q = nn.Linear(d_embed, d_s, bias=False)
        self.W_k = nn.Linear(d_embed, d_s, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        active_mask: torch.Tensor,
        active_alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, B, C, T) raw multi-band features
            adj: (C, C) binary adjacency (physical graph)
            active_mask: (batch, T, B) binary mask for active bands
            active_alpha: (batch, T, B) renormalized weights for active bands

        Returns:
            P_func: (batch, T, C, C) fused functional prior (row-stochastic on adj)
            embeddings_mean: (batch, T, d_embed) mean embedding for gate computation
        """
        batch, B, C, T = x.shape
        device = x.device

        # x rearranged to (batch, T, B, C)
        x_t = x.permute(0, 3, 1, 2)  # (batch, T, B, C)

        neg_inf = torch.finfo(x.dtype).min
        adj_mask = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, C, C)

        P_func_accum = torch.zeros(batch, T, C, C, device=device, dtype=x.dtype)
        embed_accum = torch.zeros(batch, T, self.d_embed, device=device, dtype=x.dtype)
        active_count = torch.zeros(batch, T, 1, device=device, dtype=x.dtype)

        for b in range(B):
            # Embed: (batch, T, C, 1) -> (batch, T, C, d_embed)
            e_b = self.band_embeds[b](x_t[:, :, b, :].unsqueeze(-1))  # (batch, T, C, d_embed)

            # Low-dim projections
            q = self.W_q(e_b)  # (batch, T, C, d_s)
            k = self.W_k(e_b)  # (batch, T, C, d_s)

            # Similarity: (batch, T, C, C)
            s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_s)

            # Mask to physical edges
            s = torch.where(adj_mask > 0, s, torch.full_like(s, neg_inf))

            # Row-stochastic via neighborhood softmax
            P_b = F.softmax(s, dim=-1)  # (batch, T, C, C)

            # Weight by active_alpha for this band
            w_b = active_alpha[:, :, b].unsqueeze(-1).unsqueeze(-1)  # (batch, T, 1, 1)
            mask_b = active_mask[:, :, b].unsqueeze(-1).unsqueeze(-1)  # (batch, T, 1, 1)

            P_func_accum = P_func_accum + w_b * mask_b * P_b

            # Accumulate embeddings for gate (mean over C and active bands)
            embed_accum = embed_accum + mask_b.squeeze(-1) * e_b.mean(dim=2)
            active_count = active_count + mask_b.squeeze(-1).squeeze(-1).unsqueeze(-1)

        # Mean embedding over active bands
        embeddings_mean = embed_accum / active_count.clamp(min=1)  # (batch, T, d_embed)

        return P_func_accum, embeddings_mean


class PriorMixingGate(nn.Module):
    """
    Computes g_t = sigma(w_g^T e_bar_t + b_g) (Sec 4.2, Eq. 16).
    """

    def __init__(self, d_embed: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(d_embed, 1)

    def forward(self, embeddings_mean: torch.Tensor) -> torch.Tensor:
        """
        embeddings_mean: (batch, T, d_embed)
        Returns: g_t: (batch, T, 1, 1) for broadcasting with (C, C) priors
        """
        g = torch.sigmoid(self.gate_proj(embeddings_mean))  # (batch, T, 1)
        return g.unsqueeze(-1)  # (batch, T, 1, 1)


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head graph attention with additive log-bias from prior (Sec 4.3, Eq. 14).
    l_{ij} = LeakyReLU(a^T [Wh_i || Wh_j]) + beta * log(P_{ij} + eps)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W = nn.Parameter(torch.empty(num_heads, d_model, self.d_head))
        self.a_src = nn.Parameter(torch.empty(num_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.d_head))

        # Learnable prior strength: beta = log(1 + exp(beta_tilde)) > 0
        self.beta_tilde = nn.Parameter(torch.tensor(1.0))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        self.eps = 1e-6
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.num_heads):
            nn.init.xavier_uniform_(self.W[h])
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, D)
        adj: (N, N) binary adjacency
        prior: (B, N, N) row-stochastic prior P_t, or None
        """
        B, N, D = x.shape
        H = self.num_heads

        x_proj = torch.einsum("bnd,hdf->bhnf", x, self.W)  # (B, H, N, d_head)

        e_src = (x_proj * self.a_src[None, :, None, :]).sum(-1)  # (B, H, N)
        e_dst = (x_proj * self.a_dst[None, :, None, :]).sum(-1)  # (B, H, N)

        e = e_src.unsqueeze(-1) + e_dst.unsqueeze(-2)            # (B, H, N, N)
        e = self.leakyrelu(e)

        # Add prior log-bias: beta * log(P_{ij} + eps)
        if prior is not None:
            beta = F.softplus(self.beta_tilde)  # beta > 0
            log_prior = beta * torch.log(prior + self.eps)  # (B, N, N)
            e = e + log_prior.unsqueeze(1)  # broadcast over heads

        # Mask to physical edges
        adj_exp = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        neg_inf = torch.finfo(e.dtype).min
        e = torch.where(adj_exp > 0, e, torch.full_like(e, neg_inf))

        attn = F.softmax(e, dim=-1)
        attn = self.dropout(attn)

        h_prime = torch.matmul(attn, x_proj)  # (B, H, N, d_head)
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(B, N, D)

        out = self.W_out(h_prime)
        return out


class SpatialTGALayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadGraphAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()  # Paper Eq. 18: ELU activation

    def forward(self, x_t: torch.Tensor, adj: torch.Tensor,
                prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn(x_t, adj, prior=prior)
        x_t = self.norm(x_t + self.dropout(h))
        x_t = self.act(x_t)
        return x_t


class TemporalBlock(nn.Module):
    """
    Local causal self-attention over time (CLTA, Sec 4.3, Eq. 16).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 10,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def _build_local_causal_mask(self, T: int, device, dtype) -> torch.Tensor:
        idx = torch.arange(T, device=device)
        dist = idx.view(-1, 1) - idx.view(1, -1)
        valid = (dist >= 0) & (dist < self.window_size)

        neg_inf = torch.finfo(dtype).min
        mask = torch.zeros((T, T), device=device, dtype=dtype)
        mask[~valid] = neg_inf
        return mask.view(1, 1, T, T)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        H = self.num_heads
        d_head = self.d_head

        q = self.q_proj(z).view(B, T, H, d_head).transpose(1, 2)
        k = self.k_proj(z).view(B, T, H, d_head).transpose(1, 2)
        v = self.v_proj(z).view(B, T, H, d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
        mask = self._build_local_causal_mask(T, z.device, scores.dtype)
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        z = self.norm1(z + self.dropout(out))
        ff = self.ffn(z)
        z = self.norm2(z + ff)
        return z


__all__ = [
    "CausalSpectralFusion",
    "BandFunctionalPrior",
    "PriorMixingGate",
    "MultiHeadGraphAttention",
    "SpatialTGALayer",
    "TemporalBlock",
]
