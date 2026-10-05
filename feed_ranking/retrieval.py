# Retrieval stage: a category inverted index, not a learned model, the cheapest real
# stage in the cascade. Given a user's top affinity categories, pulls candidate item ids
# from exactly those categories' posting lists and samples down to a fixed candidate-set
# size, so every request's retrieval cost is bounded regardless of the 1,000,000-item
# catalog behind it.
from __future__ import annotations

import numpy as np

from .catalog import Catalog, Users


def retrieve_candidates(
    user_index: int,
    users: Users,
    catalog: Catalog,
    rng: np.random.Generator,
    n_candidates: int = 2000,
) -> np.ndarray:
    pool_parts = [catalog.category_index[int(c)] for c in users.top_categories[user_index]]
    pool = np.concatenate(pool_parts)
    pool = np.unique(pool)
    if len(pool) <= n_candidates:
        return pool
    return rng.choice(pool, size=n_candidates, replace=False)
