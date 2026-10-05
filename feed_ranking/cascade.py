# The cascade itself: retrieval -> pre-ranker cut -> reranker, plus the one baseline the
# whole project is measured against: the same reranker applied directly to every
# retrieval candidate, skipping the pre-ranker cut entirely. That baseline is what "held
# 97% of the full reranker's NDCG@10" and "ranking compute cut 83%" are both measured
# relative to; it is never the thing actually served, it exists only to be compared
# against.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .catalog import Catalog, Users
from .reranker import Reranker
from .retrieval import retrieve_candidates
from .towers import ItemTower, UserTower, dot_scores

PRE_RANKER_CUT = 200
TOP_K = 10


@dataclass(frozen=True)
class CascadeResult:
    candidates: np.ndarray  # the full retrieval set (up to 2,000 item ids)
    pre_ranker_survivors: np.ndarray  # the 200 item ids the pre-ranker kept
    cascade_top_k: np.ndarray  # reranker's top-K among the 200 survivors
    baseline_top_k: np.ndarray  # reranker's top-K among all retrieval candidates directly


def user_embedding(user_tower: UserTower, users: Users, user_index: int) -> torch.Tensor:
    cats = torch.as_tensor(users.top_categories[user_index], dtype=torch.long).unsqueeze(0)
    weights = torch.as_tensor(users.category_weights[user_index], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return user_tower(cats, weights).squeeze(0)


def item_embeddings(item_tower: ItemTower, catalog: Catalog, item_ids: np.ndarray) -> torch.Tensor:
    cats = torch.as_tensor(catalog.item_category[item_ids], dtype=torch.long)
    with torch.no_grad():
        return item_tower(cats)


def _top_k_by_score(item_ids: np.ndarray, scores: torch.Tensor, k: int) -> np.ndarray:
    k = min(k, len(item_ids))
    order = torch.argsort(scores, descending=True)[:k].numpy()
    return item_ids[order]


def run_cascade(
    user_index: int,
    users: Users,
    catalog: Catalog,
    user_tower: UserTower,
    item_tower: ItemTower,
    reranker: Reranker,
    rng: np.random.Generator,
    *,
    n_candidates: int = 2000,
    pre_ranker_cut: int = PRE_RANKER_CUT,
    top_k: int = TOP_K,
) -> CascadeResult:
    candidates = retrieve_candidates(user_index, users, catalog, rng, n_candidates=n_candidates)
    u_emb = user_embedding(user_tower, users, user_index)
    cand_embs = item_embeddings(item_tower, catalog, candidates)

    pre_scores = dot_scores(u_emb, cand_embs)
    survivors = _top_k_by_score(candidates, pre_scores, pre_ranker_cut)

    with torch.no_grad():
        survivor_embs = item_embeddings(item_tower, catalog, survivors)
        survivor_quality = torch.as_tensor(catalog.item_quality[survivors], dtype=torch.float32)
        cascade_scores = reranker(u_emb, survivor_embs, survivor_quality)
    cascade_top_k = _top_k_by_score(survivors, cascade_scores, top_k)

    with torch.no_grad():
        cand_quality = torch.as_tensor(catalog.item_quality[candidates], dtype=torch.float32)
        baseline_scores = reranker(u_emb, cand_embs, cand_quality)
    baseline_top_k = _top_k_by_score(candidates, baseline_scores, top_k)

    return CascadeResult(
        candidates=candidates,
        pre_ranker_survivors=survivors,
        cascade_top_k=cascade_top_k,
        baseline_top_k=baseline_top_k,
    )
