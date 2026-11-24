"""OddsMoveNet: per-book temporal encoder + cross-book attention graph."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OddsMoveNet(nn.Module):
    """Predicts P(down, flat, up) for every bookmaker's next price move.

    1. A GRU shared across books summarises each book's recent tick history
       into a hidden state. A learned book embedding lets the shared encoder
       specialise to each book's pricing behaviour.
    2. Multi-head attention over the K book states acts as a learned, dynamic
       correlation graph: the attention weights say which books a given book
       is currently "listening" to (i.e. who leads whom right now).
    3. A small head maps [own state, graph message] to 3-way logits.

    A scalar temperature, fitted on validation data after training, calibrates
    the output probabilities.
    """

    def __init__(self, n_books: int, n_features: int = 3, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.n_books = n_books
        self.inp = nn.Linear(n_features, d_model)
        self.book_emb = nn.Embedding(n_books, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
        )
        self.register_buffer("temperature", torch.ones(1))
        self.register_buffer("book_ids", torch.arange(n_books))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, K, F) -> book states (B, K, 2d), attention (B, K, K)."""
        B, L, K, _ = x.shape
        h = self.inp(x) + self.book_emb(self.book_ids)  # (B, L, K, d)
        h = h.permute(0, 2, 1, 3).reshape(B * K, L, -1)
        _, hn = self.gru(h)
        e = hn[-1].view(B, K, -1)
        msg, attn = self.attn(e, e, e, need_weights=True)
        return torch.cat([e, self.norm(msg)], dim=-1), attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated logits, shape (B, K, 3)."""
        z, _ = self.encode(x)
        return self.head(z) / self.temperature

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Temperature scaling (Guo et al., 2017) on held-out raw logits."""
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
        logits, labels = logits.reshape(-1, 3).detach(), labels.reshape(-1)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / log_t.exp(), labels)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature.copy_(log_t.exp().detach())
        return float(self.temperature)
