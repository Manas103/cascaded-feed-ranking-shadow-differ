# The hidden ground truth: a latent dot-product relevance function and a grader built
# from it. Every model in this repository is trained on noisy clicks sampled FROM this
# function, never on the function itself, and NDCG@10 is graded against the same ground
# truth the clicks came from, not against either model's own output. This is what makes
# "held 97% of the full reranker's NDCG@10" a measurement rather than a tautology.
from __future__ import annotations

import math

import numpy as np

LATENT_DIM = 16
_SCALE = 1.0 / math.sqrt(LATENT_DIM)
# How much a real but category-blind quality signal moves relevance, relative to the
# embeddable dot-product term. Nonzero on purpose: the two-tower pre-ranker's item
# embedding only ever encodes category (towers.py's ItemTower), so this term is a real
# relevance driver the pre-ranker's dot product structurally cannot see, while the
# reranker's cross features (reranker.py) do see it. That asymmetry, not a tuned noise
# level, is what a pre-ranker cut can genuinely lose quality from.
QUALITY_WEIGHT = 1.4


def true_relevance(user_latent: np.ndarray, item_latents: np.ndarray, item_quality: np.ndarray | None = None) -> np.ndarray:
    """Sigmoid of a scaled dot product plus a quality term: one user latent vector against
    N item latent vectors (and, when given, N quality scalars), returns N relevance
    probabilities in (0, 1)."""
    logits = (item_latents @ user_latent) * _SCALE
    if item_quality is not None:
        logits = logits + QUALITY_WEIGHT * (item_quality - 0.5) * 2.0
    return 1.0 / (1.0 + np.exp(-logits))


def graded_relevance(
    user_latent: np.ndarray, item_latents: np.ndarray, item_quality: np.ndarray | None = None, n_grades: int = 5
) -> np.ndarray:
    """Quantizes true_relevance into integer grades 0..n_grades-1 for classic NDCG, using
    fixed, data-independent probability-space cut points (not a per-batch quantile, so the
    same relevance value always grades the same way no matter what else is in the batch)."""
    p = true_relevance(user_latent, item_latents, item_quality)
    edges = np.linspace(0.0, 1.0, n_grades + 1)[1:-1]
    return np.digitize(p, edges).astype(np.int32)


def sample_clicks(
    rng: np.random.Generator, user_latent: np.ndarray, item_latents: np.ndarray, item_quality: np.ndarray | None = None
) -> np.ndarray:
    """Bernoulli click samples at the true relevance probability: the only label any model
    in this repository is ever trained on."""
    p = true_relevance(user_latent, item_latents, item_quality)
    return (rng.random(size=p.shape) < p).astype(np.float32)


def dcg_at_k(grades_in_rank_order: np.ndarray, k: int = 10) -> float:
    grades = grades_in_rank_order[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(grades) + 2))
    return float(np.sum((2.0**grades - 1.0) * discounts))


def ndcg_at_k(ranked_grades: np.ndarray, ideal_grades_sorted_desc: np.ndarray, k: int = 10) -> float:
    dcg = dcg_at_k(ranked_grades, k)
    idcg = dcg_at_k(ideal_grades_sorted_desc, k)
    return dcg / idcg if idcg > 0 else 0.0
