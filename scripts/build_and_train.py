#!/usr/bin/env python
# Builds the full 1,000,000-item / 200,000-user simulated universe, trains the two-tower
# pre-ranker and the reranker on noisy clicks sampled from the hidden relevance function,
# and saves everything artifacts/ needs for the rest of the benchmarks. Run once; every
# other script in this directory loads from artifacts/ rather than retraining.
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feed_ranking.catalog import build_catalog, build_users
from feed_ranking.training import train_reranker, train_towers

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    t0 = time.time()
    catalog = build_catalog()
    users = build_users()
    print(f"built catalog ({len(catalog.item_category):,} items) and users ({len(users.top_categories):,}) in {time.time() - t0:.1f}s")

    # The pre-ranker gets fewer steps on purpose: it is the cheap, cheerful tier in a real
    # cascade, trained less heavily than the reranker it feeds. That gap, not a tuned
    # number, is what makes the pre-ranker's cut occasionally drop a candidate the
    # better-fit reranker would have liked, which is the real source of the NDCG loss
    # measured in measure_ndcg_and_compute.py.
    t0 = time.time()
    user_tower, item_tower = train_towers(catalog, users, steps=300, batch_users=24, items_per_user=64)
    print(f"trained two-tower pre-ranker in {time.time() - t0:.1f}s")

    t0 = time.time()
    reranker = train_reranker(catalog, users, user_tower, item_tower, steps=1200, batch_users=24, items_per_user=64, seed=2)
    print(f"trained production reranker in {time.time() - t0:.1f}s")

    # Deliberately undertrained relative to production: a realistic shadow-mode scenario
    # is validating an earlier checkpoint of a candidate model before it has converged,
    # not two copies of the same fully-trained model with different random seeds (which
    # converge to nearly identical rankings and would make the differ's job trivial).
    t0 = time.time()
    shadow_reranker = train_reranker(catalog, users, user_tower, item_tower, steps=20, batch_users=24, items_per_user=64, seed=99)
    print(f"trained shadow-candidate reranker in {time.time() - t0:.1f}s")

    torch.save(user_tower.state_dict(), ARTIFACTS / "user_tower.pt")
    torch.save(item_tower.state_dict(), ARTIFACTS / "item_tower.pt")
    torch.save(reranker.state_dict(), ARTIFACTS / "reranker.pt")
    torch.save(shadow_reranker.state_dict(), ARTIFACTS / "shadow_reranker.pt")
    print(f"saved model weights to {ARTIFACTS}")


if __name__ == "__main__":
    main()
