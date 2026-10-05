# Trains both towers and the reranker on noisy Bernoulli clicks sampled from
# relevance.true_relevance (never on true_relevance itself, and never on the latent
# vectors it reads). Every training batch is built the same way serving traffic is: pick a
# real user, retrieve real candidates through the real category index, label them with a
# real (noisy) click sample.
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .catalog import Catalog, Users
from .relevance import sample_clicks
from .reranker import Reranker
from .retrieval import retrieve_candidates
from .towers import ItemTower, UserTower


def _user_tower_inputs(users: Users, user_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    cats = torch.as_tensor(users.top_categories[user_index], dtype=torch.long).unsqueeze(0)
    weights = torch.as_tensor(users.category_weights[user_index], dtype=torch.float32).unsqueeze(0)
    return cats, weights


def _make_batch(
    catalog: Catalog, users: Users, rng: np.random.Generator, batch_users: int, items_per_user: int
) -> list[tuple[int, np.ndarray]]:
    user_indices = rng.choice(len(users.top_categories), size=batch_users, replace=False)
    batch = []
    for u in user_indices:
        candidates = retrieve_candidates(int(u), users, catalog, rng, n_candidates=items_per_user)
        batch.append((int(u), candidates))
    return batch


def train_towers(
    catalog: Catalog,
    users: Users,
    *,
    steps: int = 400,
    batch_users: int = 16,
    items_per_user: int = 64,
    lr: float = 1e-2,
    seed: int = 1,
) -> tuple[UserTower, ItemTower]:
    rng = np.random.default_rng(seed)
    user_tower, item_tower = UserTower(), ItemTower()
    opt = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        batch = _make_batch(catalog, users, rng, batch_users, items_per_user)
        opt.zero_grad()
        total_loss = torch.tensor(0.0)
        for u, candidates in batch:
            cats, weights = _user_tower_inputs(users, u)
            user_emb = user_tower(cats, weights).squeeze(0)
            item_cats = torch.as_tensor(catalog.item_category[candidates], dtype=torch.long)
            item_embs = item_tower(item_cats)
            logits = item_embs @ user_emb
            user_latent = torch.as_tensor(users.user_latent[u])
            item_latents = torch.as_tensor(catalog.item_latent[candidates])
            item_quality = catalog.item_quality[candidates]
            labels = torch.as_tensor(sample_clicks(rng, user_latent.numpy(), item_latents.numpy(), item_quality))
            total_loss = total_loss + loss_fn(logits, labels)
        (total_loss / len(batch)).backward()
        opt.step()
    return user_tower, item_tower


def train_reranker(
    catalog: Catalog,
    users: Users,
    user_tower: UserTower,
    item_tower: ItemTower,
    *,
    steps: int = 400,
    batch_users: int = 16,
    items_per_user: int = 64,
    lr: float = 1e-2,
    seed: int = 2,
) -> Reranker:
    rng = np.random.default_rng(seed)
    reranker = Reranker()
    opt = torch.optim.Adam(reranker.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    user_tower.eval()
    item_tower.eval()

    for _ in range(steps):
        batch = _make_batch(catalog, users, rng, batch_users, items_per_user)
        opt.zero_grad()
        total_loss = torch.tensor(0.0)
        for u, candidates in batch:
            with torch.no_grad():
                cats, weights = _user_tower_inputs(users, u)
                user_emb = user_tower(cats, weights).squeeze(0)
                item_cats = torch.as_tensor(catalog.item_category[candidates], dtype=torch.long)
                item_embs = item_tower(item_cats)
            item_quality_t = torch.as_tensor(catalog.item_quality[candidates], dtype=torch.float32)
            logits = reranker(user_emb, item_embs, item_quality_t)
            user_latent = torch.as_tensor(users.user_latent[u])
            item_latents = torch.as_tensor(catalog.item_latent[candidates])
            labels = torch.as_tensor(
                sample_clicks(rng, user_latent.numpy(), item_latents.numpy(), catalog.item_quality[candidates])
            )
            total_loss = total_loss + loss_fn(logits, labels)
        (total_loss / len(batch)).backward()
        opt.step()
    return reranker
