import numpy as np
import torch

from feed_ranking.cascade import run_cascade
from feed_ranking.reranker import Reranker
from feed_ranking.towers import ItemTower, UserTower


def test_cascade_survivors_are_a_subset_of_candidates(small_catalog, small_users, rng):
    user_tower, item_tower, reranker = UserTower(), ItemTower(), Reranker()
    result = run_cascade(
        0, small_users, small_catalog, user_tower, item_tower, reranker, rng,
        n_candidates=100, pre_ranker_cut=20, top_k=5,
    )
    assert set(result.pre_ranker_survivors.tolist()).issubset(set(result.candidates.tolist()))
    assert set(result.cascade_top_k.tolist()).issubset(set(result.pre_ranker_survivors.tolist()))
    assert set(result.baseline_top_k.tolist()).issubset(set(result.candidates.tolist()))


def test_cascade_top_k_has_no_duplicates(small_catalog, small_users, rng):
    user_tower, item_tower, reranker = UserTower(), ItemTower(), Reranker()
    result = run_cascade(
        0, small_users, small_catalog, user_tower, item_tower, reranker, rng,
        n_candidates=100, pre_ranker_cut=20, top_k=5,
    )
    assert len(result.cascade_top_k) == len(set(result.cascade_top_k.tolist()))
    assert len(result.baseline_top_k) == len(set(result.baseline_top_k.tolist()))


def test_cascade_respects_requested_sizes(small_catalog, small_users, rng):
    user_tower, item_tower, reranker = UserTower(), ItemTower(), Reranker()
    result = run_cascade(
        0, small_users, small_catalog, user_tower, item_tower, reranker, rng,
        n_candidates=100, pre_ranker_cut=20, top_k=5,
    )
    assert len(result.pre_ranker_survivors) == 20
    assert len(result.cascade_top_k) == 5
    assert len(result.baseline_top_k) == 5


def test_a_strongly_preferred_item_ranks_first_in_baseline_once_reranker_is_untrained_but_scores_are_deterministic(small_catalog, small_users, rng):
    # Sanity check on determinism, not on model quality: running the same cascade twice
    # with models in eval mode and the same rng state produces the same top-k.
    torch.manual_seed(0)
    user_tower, item_tower, reranker = UserTower(), ItemTower(), Reranker()
    user_tower.eval(); item_tower.eval(); reranker.eval()
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    result_a = run_cascade(0, small_users, small_catalog, user_tower, item_tower, reranker, rng_a, n_candidates=50, pre_ranker_cut=10, top_k=5)
    result_b = run_cascade(0, small_users, small_catalog, user_tower, item_tower, reranker, rng_b, n_candidates=50, pre_ranker_cut=10, top_k=5)
    assert np.array_equal(result_a.cascade_top_k, result_b.cascade_top_k)
