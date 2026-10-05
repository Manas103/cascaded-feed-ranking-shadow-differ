import numpy as np

from feed_ranking.shadow_differ import fast_diff, reference_diff, run_shadow_diff


def test_identical_rankings_are_not_flagged():
    a = np.array([5, 3, 1, 9])
    assert fast_diff(a, a.copy()) is False
    assert reference_diff(a, a.copy()) is False


def test_a_single_swapped_pair_is_flagged():
    a = np.array([5, 3, 1, 9])
    b = np.array([3, 5, 1, 9])
    assert fast_diff(a, b) is True
    assert reference_diff(a, b) is True


def test_a_completely_different_ranking_is_flagged():
    a = np.array([1, 2, 3])
    b = np.array([9, 8, 7])
    assert fast_diff(a, b) is True
    assert reference_diff(a, b) is True


def test_fast_and_reference_agree_on_a_random_batch():
    rng = np.random.default_rng(3)
    for _ in range(200):
        base = rng.choice(1000, size=10, replace=False)
        other = base.copy()
        if rng.random() < 0.5:
            i, j = rng.choice(10, size=2, replace=False)
            other[i], other[j] = other[j], other[i]
        assert fast_diff(base, other) == reference_diff(base, other)


def test_run_shadow_diff_reports_exact_agreement_and_correct_flag_count():
    rankings_a = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    rankings_b = [np.array([1, 2, 3]), np.array([5, 4, 6]), np.array([7, 8, 9])]
    report = run_shadow_diff(rankings_a, rankings_b)
    assert report.n_requests == 3
    assert report.n_flagged_divergent == 1
    assert report.fast_and_reference_agree_on_every_request is True
    assert report.mismatched_request_indices == []
