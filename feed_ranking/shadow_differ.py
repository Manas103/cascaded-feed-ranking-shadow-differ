# Shadow mode: before a new reranker ships, replay real requests through the current
# ("production") reranker and a candidate ("shadow") reranker side by side, over the same
# retrieval candidates and the same pre-ranker survivors, and flag every request where the
# two final top-K orderings differ. Two independent implementations of the same diff
# decision (fast_diff and reference_diff below) are run over every replayed request and
# must agree exactly; that agreement, not either one alone, is what proves the differ
# itself has no off-by-one or short-circuit bug hiding a real divergence.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fast_diff(ranking_a: np.ndarray, ranking_b: np.ndarray) -> bool:
    """The production comparison: are the two rankings the same sequence, element for
    element."""
    return not np.array_equal(ranking_a, ranking_b)


def reference_diff(ranking_a: np.ndarray, ranking_b: np.ndarray) -> bool:
    """An independently coded comparison: same decision, different mechanism (a
    position-indexed dict walk rather than a vectorized array comparison), so a bug
    specific to one implementation cannot silently agree with the other by construction."""
    a, b = list(ranking_a), list(ranking_b)
    if len(a) != len(b):
        return True
    position_of_a = {int(item): i for i, item in enumerate(a)}
    for i, item in enumerate(b):
        if position_of_a.get(int(item), -1) != i:
            return True
    return False


@dataclass(frozen=True)
class DiffReport:
    n_requests: int
    n_flagged_divergent: int
    fast_and_reference_agree_on_every_request: bool
    mismatched_request_indices: list[int]


def run_shadow_diff(rankings_a: list[np.ndarray], rankings_b: list[np.ndarray]) -> DiffReport:
    assert len(rankings_a) == len(rankings_b)
    flagged = 0
    mismatches = []
    for i, (a, b) in enumerate(zip(rankings_a, rankings_b)):
        fast = fast_diff(a, b)
        reference = reference_diff(a, b)
        if fast != reference:
            mismatches.append(i)
        if fast:
            flagged += 1
    return DiffReport(
        n_requests=len(rankings_a),
        n_flagged_divergent=flagged,
        fast_and_reference_agree_on_every_request=len(mismatches) == 0,
        mismatched_request_indices=mismatches,
    )
