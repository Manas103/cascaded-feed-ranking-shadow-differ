# The simulated universe: a 1,000,000-item catalog and a 200,000-user base. Every item
# gets a category and a hidden latent preference vector; every user gets a handful of
# affinity categories and a hidden latent vector. The hidden latents are used only by
# relevance.py to grade ground truth and generate noisy training clicks; no model anywhere
# in this repository ever reads them directly, the same separation a real offline-label
# generator keeps from the online features a model actually serves on.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_ITEMS = 1_000_000
NUM_USERS = 200_000
NUM_CATEGORIES = 200
LATENT_DIM = 16
USER_TOP_CATEGORIES = 5


@dataclass
class Catalog:
    item_category: np.ndarray  # (NUM_ITEMS,) int32
    item_latent: np.ndarray  # (NUM_ITEMS, LATENT_DIM) float32, hidden from every model
    item_quality: np.ndarray  # (NUM_ITEMS,) float32 in [0, 1]; a real relevance driver the
    # category-only two-tower embeddings never see, only the reranker's cross features do
    category_index: dict  # category_id -> np.ndarray of item ids in that category


@dataclass
class Users:
    top_categories: np.ndarray  # (NUM_USERS, USER_TOP_CATEGORIES) int32
    category_weights: np.ndarray  # (NUM_USERS, USER_TOP_CATEGORIES) float32, sums to 1 per row
    user_latent: np.ndarray  # (NUM_USERS, LATENT_DIM) float32, hidden from every model


def build_catalog(seed: int = 20261004, num_items: int = NUM_ITEMS, num_categories: int = NUM_CATEGORIES) -> Catalog:
    rng = np.random.default_rng(seed)
    item_category = rng.integers(0, num_categories, size=num_items, dtype=np.int32)
    item_latent = rng.normal(0.0, 1.0, size=(num_items, LATENT_DIM)).astype(np.float32)
    item_quality = rng.random(size=num_items).astype(np.float32)
    category_index = {
        cat: np.flatnonzero(item_category == cat).astype(np.int64) for cat in range(num_categories)
    }
    return Catalog(
        item_category=item_category, item_latent=item_latent, item_quality=item_quality, category_index=category_index
    )


def build_users(
    seed: int = 20261005,
    num_users: int = NUM_USERS,
    num_categories: int = NUM_CATEGORIES,
    top_k: int = USER_TOP_CATEGORIES,
) -> Users:
    rng = np.random.default_rng(seed)
    top_categories = np.stack(
        [rng.choice(num_categories, size=top_k, replace=False) for _ in range(num_users)]
    ).astype(np.int32)
    raw_weights = rng.dirichlet(np.ones(top_k) * 2.0, size=num_users).astype(np.float32)
    user_latent = rng.normal(0.0, 1.0, size=(num_users, LATENT_DIM)).astype(np.float32)
    return Users(top_categories=top_categories, category_weights=raw_weights, user_latent=user_latent)
