#!/usr/bin/env python
# The two headline tradeoff numbers, measured together because they are one tradeoff:
# how much ranking compute the pre-ranker cut saves, and how much of the full reranker's
# NDCG@10 the cascade still delivers after that cut. Loads the models
# scripts/build_and_train.py already trained; trains nothing itself.
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feed_ranking.cascade import PRE_RANKER_CUT, item_embeddings, run_cascade, user_embedding
from feed_ranking.catalog import build_catalog, build_users
from feed_ranking.reranker import Reranker
from feed_ranking.relevance import graded_relevance, ndcg_at_k
from feed_ranking.retrieval import retrieve_candidates
from feed_ranking.towers import ItemTower, UserTower

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_QUERY_USERS = 2000
N_CANDIDATES = 2000
SEED = 20261004


def load_models() -> tuple[UserTower, ItemTower, Reranker]:
    user_tower, item_tower, reranker = UserTower(), ItemTower(), Reranker()
    user_tower.load_state_dict(torch.load(ARTIFACTS / "user_tower.pt", weights_only=True))
    item_tower.load_state_dict(torch.load(ARTIFACTS / "item_tower.pt", weights_only=True))
    reranker.load_state_dict(torch.load(ARTIFACTS / "reranker.pt", weights_only=True))
    user_tower.eval(); item_tower.eval(); reranker.eval()
    return user_tower, item_tower, reranker


def measure_ndcg(catalog, users, user_tower, item_tower, reranker, rng) -> dict:
    query_users = rng.choice(len(users.top_categories), size=N_QUERY_USERS, replace=False)
    cascade_ndcgs, baseline_ndcgs = [], []
    survivor_sizes, candidate_sizes = [], []

    for u in query_users:
        result = run_cascade(
            int(u), users, catalog, user_tower, item_tower, reranker, rng,
            n_candidates=N_CANDIDATES, pre_ranker_cut=PRE_RANKER_CUT, top_k=10,
        )
        candidate_sizes.append(len(result.candidates))
        survivor_sizes.append(len(result.pre_ranker_survivors))

        user_latent = users.user_latent[int(u)]
        all_grades = graded_relevance(
            user_latent, catalog.item_latent[result.candidates], catalog.item_quality[result.candidates]
        )
        grade_lookup = dict(zip(result.candidates.tolist(), all_grades.tolist()))
        ideal_sorted = np.sort(all_grades)[::-1]

        cascade_grades = np.array([grade_lookup[i] for i in result.cascade_top_k.tolist()])
        baseline_grades = np.array([grade_lookup[i] for i in result.baseline_top_k.tolist()])
        cascade_ndcgs.append(ndcg_at_k(cascade_grades, ideal_sorted, k=10))
        baseline_ndcgs.append(ndcg_at_k(baseline_grades, ideal_sorted, k=10))

    mean_cascade = float(np.mean(cascade_ndcgs))
    mean_baseline = float(np.mean(baseline_ndcgs))
    return {
        "n_query_users": N_QUERY_USERS,
        "mean_candidates_per_request": float(np.mean(candidate_sizes)),
        "mean_pre_ranker_survivors_per_request": float(np.mean(survivor_sizes)),
        "mean_cascade_ndcg_at_10": mean_cascade,
        "mean_baseline_full_reranker_ndcg_at_10": mean_baseline,
        "pct_of_baseline_ndcg_held": round(100 * mean_cascade / mean_baseline, 2),
    }


def measure_compute_reduction(catalog, users, user_tower, item_tower, reranker, rng, *, repeats: int = 20) -> dict:
    """Wall-clock compute for two paths, averaged over `repeats` requests, each timed in
    isolation with torch.no_grad() and the model already warmed up: (a) baseline, the
    reranker scoring all 2,000 retrieval candidates directly; (b) cascade, the pre-ranker's
    cached-embedding dot product over all 2,000 plus the reranker scoring only the 200
    survivors. The pre-ranker's own dot-product cost is counted in (b), not treated as
    free, which is what keeps this an honest compute comparison rather than a raw
    item-count ratio."""
    query_users = rng.choice(len(users.top_categories), size=repeats, replace=False)

    # Warm up (first-call overhead, lazy CUDA/BLAS init, etc. should not count).
    warm_u = int(query_users[0])
    candidates = retrieve_candidates(warm_u, users, catalog, rng, n_candidates=N_CANDIDATES)
    u_emb = user_embedding(user_tower, users, warm_u)
    cand_embs = item_embeddings(item_tower, catalog, candidates)
    cand_quality = torch.as_tensor(catalog.item_quality[candidates], dtype=torch.float32)
    with torch.no_grad():
        _ = reranker(u_emb, cand_embs, cand_quality)

    baseline_times, cascade_times = [], []
    for u in query_users:
        u = int(u)
        candidates = retrieve_candidates(u, users, catalog, rng, n_candidates=N_CANDIDATES)
        u_emb = user_embedding(user_tower, users, u)
        cand_embs = item_embeddings(item_tower, catalog, candidates)
        cand_quality = torch.as_tensor(catalog.item_quality[candidates], dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = reranker(u_emb, cand_embs, cand_quality)
        baseline_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        from feed_ranking.towers import dot_scores
        pre_scores = dot_scores(u_emb, cand_embs)
        order = torch.argsort(pre_scores, descending=True)[:PRE_RANKER_CUT]
        survivor_embs = cand_embs[order]
        survivor_quality = cand_quality[order]
        with torch.no_grad():
            _ = reranker(u_emb, survivor_embs, survivor_quality)
        cascade_times.append(time.perf_counter() - t0)

    mean_baseline = float(np.mean(baseline_times))
    mean_cascade = float(np.mean(cascade_times))
    return {
        "repeats": repeats,
        "mean_baseline_seconds_per_request": mean_baseline,
        "mean_cascade_seconds_per_request": mean_cascade,
        "pct_ranking_compute_cut": round(100 * (1 - mean_cascade / mean_baseline), 2),
    }


def main() -> None:
    # Pinned to 1 thread: PyTorch's default intra-op thread pool adds scheduling overhead
    # that is itself noisy at these tiny tensor sizes and would otherwise dominate the
    # very difference this benchmark is trying to measure.
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    catalog = build_catalog()
    users = build_users()
    user_tower, item_tower, reranker = load_models()

    ndcg_report = measure_ndcg(catalog, users, user_tower, item_tower, reranker, rng)
    compute_report = measure_compute_reduction(catalog, users, user_tower, item_tower, reranker, rng, repeats=500)

    print(json.dumps({"ndcg": ndcg_report, "compute": compute_report}, indent=2))


if __name__ == "__main__":
    main()
