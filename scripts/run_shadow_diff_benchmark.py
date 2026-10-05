#!/usr/bin/env python
# Shadow mode over 250,000 replayed requests: the production reranker (reranker.pt) and a
# shadow candidate (shadow_reranker.pt, trained with a different seed and more steps, the
# kind of change a real team would want to validate before shipping) both score the exact
# same retrieval candidates and pre-ranker survivors for the exact same request, and the
# differ flags every request where their final top-10 orderings disagree. The claim is
# that the differ flags EVERY divergence, which this script proves the only way that is
# convincing: fast_diff and reference_diff (two independently coded comparisons) must
# agree on all 250,000 decisions.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feed_ranking.cascade import PRE_RANKER_CUT, item_embeddings, user_embedding
from feed_ranking.catalog import build_catalog, build_users
from feed_ranking.reranker import Reranker
from feed_ranking.retrieval import retrieve_candidates
from feed_ranking.shadow_differ import run_shadow_diff
from feed_ranking.towers import ItemTower, UserTower, dot_scores

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_REQUESTS = 250_000
N_CANDIDATES = 2000
TOP_K = 10
SEED = 20261004


def load(name: str, cls):
    model = cls()
    model.load_state_dict(torch.load(ARTIFACTS / name, weights_only=True))
    model.eval()
    return model


def rank_top_k(item_ids: np.ndarray, scores: torch.Tensor, k: int) -> np.ndarray:
    order = torch.argsort(scores, descending=True)[:k].numpy()
    return item_ids[order]


def main() -> None:
    torch.set_num_threads(1)
    catalog = build_catalog()
    users = build_users()
    user_tower = load("user_tower.pt", UserTower)
    item_tower = load("item_tower.pt", ItemTower)
    production = load("reranker.pt", Reranker)
    shadow = load("shadow_reranker.pt", Reranker)

    rng = np.random.default_rng(SEED)
    n_users = len(users.top_categories)
    # 250,000 requests drawn with replacement across the 200,000-user base (a user can be
    # replayed more than once with an independently sampled candidate set each time, the
    # same way a real shadow-mode replay samples live traffic, not a once-per-user cap).
    query_users = rng.integers(0, n_users, size=N_REQUESTS)

    production_rankings: list[np.ndarray] = []
    shadow_rankings: list[np.ndarray] = []

    for u in query_users:
        u = int(u)
        candidates = retrieve_candidates(u, users, catalog, rng, n_candidates=N_CANDIDATES)
        u_emb = user_embedding(user_tower, users, u)
        cand_embs = item_embeddings(item_tower, catalog, candidates)
        cand_quality = torch.as_tensor(catalog.item_quality[candidates], dtype=torch.float32)

        pre_scores = dot_scores(u_emb, cand_embs)
        order = torch.argsort(pre_scores, descending=True)[:PRE_RANKER_CUT]
        survivors = candidates[order.numpy()]
        survivor_embs = cand_embs[order]
        survivor_quality = cand_quality[order]

        with torch.no_grad():
            prod_scores = production(u_emb, survivor_embs, survivor_quality)
            shadow_scores = shadow(u_emb, survivor_embs, survivor_quality)
        production_rankings.append(rank_top_k(survivors, prod_scores, TOP_K))
        shadow_rankings.append(rank_top_k(survivors, shadow_scores, TOP_K))

    report = run_shadow_diff(production_rankings, shadow_rankings)
    print(
        json.dumps(
            {
                "n_requests": report.n_requests,
                "n_flagged_divergent": report.n_flagged_divergent,
                "pct_requests_with_divergent_ordering": round(100 * report.n_flagged_divergent / report.n_requests, 2),
                "fast_and_reference_agree_on_every_request": report.fast_and_reference_agree_on_every_request,
                "n_mismatched_requests": len(report.mismatched_request_indices),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
