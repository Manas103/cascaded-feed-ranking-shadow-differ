import numpy as np

from feed_ranking.retrieval import retrieve_candidates


def test_catalog_category_index_covers_every_item(small_catalog):
    total = sum(len(ids) for ids in small_catalog.category_index.values())
    assert total == len(small_catalog.item_category)


def test_every_item_in_a_category_bucket_is_actually_that_category(small_catalog):
    for cat, ids in small_catalog.category_index.items():
        assert np.all(small_catalog.item_category[ids] == cat)


def test_user_category_weights_sum_to_one(small_users):
    sums = small_users.category_weights.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_retrieve_candidates_returns_items_from_users_own_categories(small_catalog, small_users, rng):
    for user_index in (0, 1, 50):
        candidates = retrieve_candidates(user_index, small_users, small_catalog, rng, n_candidates=100)
        allowed_categories = set(int(c) for c in small_users.top_categories[user_index])
        got_categories = set(int(c) for c in small_catalog.item_category[candidates])
        assert got_categories.issubset(allowed_categories)


def test_retrieve_candidates_never_returns_duplicates(small_catalog, small_users, rng):
    candidates = retrieve_candidates(0, small_users, small_catalog, rng, n_candidates=50)
    assert len(candidates) == len(set(candidates.tolist()))


def test_retrieve_candidates_caps_at_requested_size(small_catalog, small_users, rng):
    candidates = retrieve_candidates(0, small_users, small_catalog, rng, n_candidates=10)
    assert len(candidates) <= 10
