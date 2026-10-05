import numpy as np

from feed_ranking.relevance import dcg_at_k, graded_relevance, ndcg_at_k, true_relevance


def test_true_relevance_is_between_0_and_1(small_catalog, small_users):
    user_latent = small_users.user_latent[0]
    p = true_relevance(user_latent, small_catalog.item_latent[:200])
    assert np.all(p > 0.0) and np.all(p < 1.0)


def test_identical_latents_score_higher_than_orthogonal_ones():
    user_latent = np.ones(16, dtype=np.float32)
    aligned = np.ones((1, 16), dtype=np.float32)
    orthogonal = np.zeros((1, 16), dtype=np.float32)
    orthogonal[0, ::2] = 1.0
    orthogonal[0, 1::2] = -1.0
    p_aligned = true_relevance(user_latent, aligned)
    p_orthogonal = true_relevance(user_latent, orthogonal)
    assert p_aligned[0] > p_orthogonal[0]


def test_graded_relevance_is_bounded_by_n_grades(small_catalog, small_users):
    grades = graded_relevance(small_users.user_latent[0], small_catalog.item_latent, n_grades=5)
    assert grades.min() >= 0
    assert grades.max() <= 4


def test_ndcg_of_the_ideal_ranking_against_itself_is_1():
    ideal = np.array([4, 3, 2, 1, 0])
    assert abs(ndcg_at_k(ideal, ideal, k=5) - 1.0) < 1e-9


def test_ndcg_penalizes_a_reversed_ranking():
    ideal = np.array([4, 3, 2, 1, 0])
    reversed_order = np.array([0, 1, 2, 3, 4])
    assert ndcg_at_k(reversed_order, ideal, k=5) < 1.0


def test_dcg_only_uses_top_k():
    grades = np.array([5, 4, 3, 2, 1])
    assert dcg_at_k(grades, k=2) == dcg_at_k(grades[:2], k=2)


def test_ndcg_is_0_when_ideal_has_no_relevance():
    zero_grades = np.zeros(5, dtype=np.int32)
    assert ndcg_at_k(zero_grades, zero_grades, k=5) == 0.0
