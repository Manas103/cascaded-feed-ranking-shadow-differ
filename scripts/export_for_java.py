#!/usr/bin/env python
# Exports everything the Java serving tier needs to run the real cascade without any
# PyTorch or JSON-parsing dependency: the user tower and reranker weights as plain
# whitespace-delimited text matrices (small enough to hand-port the forward pass to
# arithmetic in Java), and, for a benchmark sample of query users, their raw tower inputs
# plus a precomputed 2,000-item retrieval candidate set with cached item embeddings and
# quality scalars (the serving-time "Redis cache" this benchmark stands in for; see
# serving/README.md's Redis section).
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feed_ranking.cascade import item_embeddings
from feed_ranking.catalog import build_catalog, build_users
from feed_ranking.reranker import Reranker
from feed_ranking.retrieval import retrieve_candidates
from feed_ranking.towers import ItemTower, UserTower

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
EXPORT_DIR = Path(__file__).resolve().parents[1] / "export"
N_BENCHMARK_USERS = 50
N_CANDIDATES = 2000
SEED = 20261004


def write_linear_stack(path: Path, state_dict: dict, layer_names: list[str]) -> None:
    with path.open("w") as f:
        f.write(f"{len(layer_names)}\n")
        for name in layer_names:
            weight = state_dict[f"{name}.weight"].numpy()
            bias = state_dict[f"{name}.bias"].numpy()
            f.write(f"{weight.shape[0]} {weight.shape[1]}\n")
            for row in weight:
                f.write(" ".join(f"{x:.8f}" for x in row) + "\n")
            f.write(" ".join(f"{x:.8f}" for x in bias) + "\n")


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(f"{matrix.shape[0]} {matrix.shape[1]}\n")
        for row in matrix:
            f.write(" ".join(f"{x:.8f}" for x in row) + "\n")


def main() -> None:
    EXPORT_DIR.mkdir(exist_ok=True)
    catalog = build_catalog()
    users = build_users()

    user_tower = UserTower()
    item_tower = ItemTower()
    reranker = Reranker()
    user_tower.load_state_dict(torch.load(ARTIFACTS / "user_tower.pt", weights_only=True))
    item_tower.load_state_dict(torch.load(ARTIFACTS / "item_tower.pt", weights_only=True))
    reranker.load_state_dict(torch.load(ARTIFACTS / "reranker.pt", weights_only=True))
    user_tower.eval(); item_tower.eval(); reranker.eval()

    sd = user_tower.state_dict()
    write_matrix(EXPORT_DIR / "user_tower_category_embedding.txt", sd["category_emb.weight"].numpy())
    write_linear_stack(EXPORT_DIR / "user_tower_mlp.txt", sd, ["mlp.0", "mlp.2"])

    sd = reranker.state_dict()
    write_linear_stack(EXPORT_DIR / "reranker_mlp.txt", sd, ["net.0", "net.2", "net.4"])

    rng = np.random.default_rng(SEED)
    n_users = len(users.top_categories)
    benchmark_users = rng.choice(n_users, size=N_BENCHMARK_USERS, replace=False)

    # Embeddings are cached per the pre-ranker's whole design (see towers.py); a real
    # service would read them from Redis keyed by item id. Exported as a flat,
    # line-per-(request, item) text file: at 50 requests x 2,000 candidates x 32 floats,
    # this parses far faster in Java and is far smaller on disk than a nested format would
    # be.
    requests_lines = []
    with (EXPORT_DIR / "candidate_embeddings.tsv").open("w") as emb_file:
        for req_index, u in enumerate(benchmark_users):
            u = int(u)
            candidates = retrieve_candidates(u, users, catalog, rng, n_candidates=N_CANDIDATES)
            with torch.no_grad():
                cand_embs = item_embeddings(item_tower, catalog, candidates).numpy()
            cand_quality = catalog.item_quality[candidates]
            for item_id, emb, quality in zip(candidates.tolist(), cand_embs, cand_quality.tolist()):
                emb_file.write(f"{req_index}\t{item_id}\t{quality:.6f}\t" + " ".join(f"{x:.6f}" for x in emb) + "\n")

            cats = " ".join(str(c) for c in users.top_categories[u].tolist())
            weights = " ".join(f"{w:.8f}" for w in users.category_weights[u].tolist())
            ids = " ".join(str(i) for i in candidates.tolist())
            requests_lines.append(f"{req_index}\t{u}\t{cats}\t{weights}\t{ids}")

    (EXPORT_DIR / "benchmark_requests.tsv").write_text("\n".join(requests_lines) + "\n")
    print(f"exported user tower, reranker, and {len(requests_lines)} benchmark requests to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
