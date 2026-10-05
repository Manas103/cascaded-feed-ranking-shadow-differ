# The pre-ranker: a two-tower model scored by a plain dot product, the cheap tier.
# Item embeddings are meant to be computed once offline and cached (ItemTower never runs
# at request time in the cascade; cascade.py only ever calls it during export/training);
# UserTower runs once per request. This is what makes the pre-ranker cheap relative to the
# reranker in compute.py's measured comparison: a cached-embedding dot product per item,
# not a per-pair forward pass.
from __future__ import annotations

import torch
from torch import nn

NUM_CATEGORIES = 200
EMB_DIM = 32


class UserTower(nn.Module):
    def __init__(self, num_categories: int = NUM_CATEGORIES, emb_dim: int = EMB_DIM) -> None:
        super().__init__()
        self.category_emb = nn.EmbeddingBag(num_categories, emb_dim, mode="sum")
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, top_categories: torch.Tensor, category_weights: torch.Tensor) -> torch.Tensor:
        pooled = self.category_emb(top_categories, per_sample_weights=category_weights)
        return self.mlp(pooled)


class ItemTower(nn.Module):
    def __init__(self, num_categories: int = NUM_CATEGORIES, emb_dim: int = EMB_DIM) -> None:
        super().__init__()
        self.category_emb = nn.Embedding(num_categories, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, item_category: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.category_emb(item_category))


def dot_scores(user_emb: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
    """user_emb: (D,), item_embs: (N, D) -> (N,) plain dot product, the pre-ranker's only
    per-item serving cost once item_embs is a cached lookup."""
    return item_embs @ user_emb
