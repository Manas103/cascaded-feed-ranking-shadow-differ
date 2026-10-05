# The reranker: the heavier tier. Takes a full per-pair cross feature (user embedding,
# item embedding, and their elementwise product, the cross term the pre-ranker's plain
# dot product never computes) through a real two-layer MLP, run fresh for every
# (user, item) pair at request time; this per-pair forward pass, not just "more
# parameters", is what makes it genuinely heavier than the pre-ranker's cached-embedding
# dot product.
from __future__ import annotations

import torch
from torch import nn

EMB_DIM = 32
HIDDEN_DIM = 256


class Reranker(nn.Module):
    """Input is the full per-pair cross feature set (item embedding, broadcast user
    embedding, their elementwise product) PLUS one scalar the pre-ranker's dot product
    never sees at all: a per-item quality signal (relevance.QUALITY_WEIGHT) that the
    category-only two-tower embeddings cannot encode. That extra, genuinely-informative
    input is what makes the reranker's ranking real-ly diverge from the pre-ranker's,
    rather than just being a more expensive way to recompute the same ranking."""

    def __init__(self, emb_dim: int = EMB_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def cross_features(self, user_emb: torch.Tensor, item_embs: torch.Tensor, item_quality: torch.Tensor) -> torch.Tensor:
        """user_emb: (D,), item_embs: (N, D), item_quality: (N,) -> (N, 3D+1)."""
        n = item_embs.shape[0]
        user_broadcast = user_emb.unsqueeze(0).expand(n, -1)
        cross = user_broadcast * item_embs
        return torch.cat([user_broadcast, item_embs, cross, item_quality.unsqueeze(1)], dim=1)

    def forward(self, user_emb: torch.Tensor, item_embs: torch.Tensor, item_quality: torch.Tensor) -> torch.Tensor:
        features = self.cross_features(user_emb, item_embs, item_quality)
        return self.net(features).squeeze(-1)
