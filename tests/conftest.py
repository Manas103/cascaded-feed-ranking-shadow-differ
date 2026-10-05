import numpy as np
import pytest

from feed_ranking.catalog import build_catalog, build_users

SMALL_ITEMS = 2000
SMALL_CATEGORIES = 20
SMALL_USERS = 200


@pytest.fixture
def small_catalog():
    return build_catalog(seed=1, num_items=SMALL_ITEMS, num_categories=SMALL_CATEGORIES)


@pytest.fixture
def small_users():
    return build_users(seed=2, num_users=SMALL_USERS, num_categories=SMALL_CATEGORIES, top_k=5)


@pytest.fixture
def rng():
    return np.random.default_rng(42)
