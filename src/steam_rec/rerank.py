"""Diversity re-ranking of the ranker's output (decisions D2, D26).

The ranker's top 10 is 75% blockbusters (D25). Two simple fixes, each tried at a few strengths:
- popularity penalty: final = relevance - ALPHA x popularity, both scaled to 0-1
- MMR (maximal marginal relevance): build the top 10 one pick at a time; each pick maximises
  LAMBDA x relevance - (1 - LAMBDA) x (its highest tag similarity to the games already picked)

Scored on the ranker's 10k held-out val users: Recall@10, NDCG@10 (all and new games),
coverage, head share, and intra-list diversity (how different the 10 games' tags are).
For each method the chosen strength is the one with the best diversity among settings that lose
at most MAX_NDCG_LOSS of the ranker's NDCG@10 (rule fixed before looking at the results).

Output: metrics/rerank_val.json.
Run with: uv run python -m steam_rec.rerank
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from steam_rec.candidates import keep_top_n
from steam_rec.content import build_tag_matrix
from steam_rec.data import PROJECT_ROOT
from steam_rec.evaluate import beyond_accuracy, ranking_metrics
from steam_rec.prepare import PROCESSED_DIR
from steam_rec.ranker import RANKED_PATH
from steam_rec.split import SPLIT_DIR, VAL_CUTOFF

METRICS_PATH = PROJECT_ROOT / "metrics" / "rerank_val.json"

FINAL_K = 10
MMR_POOL = 50
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5]
LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.5]
MAX_NDCG_LOSS = 0.05


def scale_scores(recs: pd.DataFrame) -> pd.Series:
    """Rescale each user's `score` to 0-1 (best candidate = 1, worst = 0); all-equal scores become 1."""
    user_min = recs.groupby("user_id")["score"].transform("min")
    user_max = recs.groupby("user_id")["score"].transform("max")
    spread = user_max - user_min
    scaled = (recs["score"] - user_min) / spread.where(spread > 0, 1.0)
    return scaled.where(spread > 0, 1.0)


def popularity_penalty(recs: pd.DataFrame, alpha: float, max_log_likes: float) -> pd.DataFrame:
    """Re-score with relevance - alpha x popularity (log likes scaled to 0-1) and keep each user's top FINAL_K."""
    rescored = recs[["user_id", "app_id"]].copy()
    popularity = recs["log_likes"] / max_log_likes
    rescored["score"] = scale_scores(recs) - alpha * popularity
    return keep_top_n(rescored, FINAL_K)


def mmr_select(relevance: np.ndarray, similarity: np.ndarray, lam: float, k: int) -> list[int]:
    """Greedy MMR for one user: positions of the k picks in order. `similarity` is candidates x candidates."""
    n = len(relevance)
    picked = []
    # Each candidate's highest similarity to anything picked so far (nothing picked yet: 0).
    max_similarity = np.zeros(n)
    available = np.ones(n, dtype=bool)
    for _ in range(min(k, n)):
        value = lam * relevance - (1 - lam) * max_similarity
        value[~available] = -np.inf
        best = int(np.argmax(value))  # ties go to the earlier, higher-ranked candidate
        picked.append(best)
        available[best] = False
        max_similarity = np.maximum(max_similarity, similarity[:, best])
    return picked


def tag_vectors(app_ids: np.ndarray, tfidf: np.ndarray, row_of_app: pd.Series) -> np.ndarray:
    """Each game's TF-IDF tag vector (length 1, or all zeros without tags), in the order of `app_ids`."""
    rows = row_of_app.reindex(app_ids).to_numpy()
    return tfidf[rows]


def mmr_rerank(recs: pd.DataFrame, tfidf: np.ndarray, row_of_app: pd.Series, lam: float) -> pd.DataFrame:
    """Apply MMR to each user's top MMR_POOL candidates; returns user_id, app_id, rank for the top FINAL_K."""
    pool = recs[recs["rank"] <= MMR_POOL].sort_values(["user_id", "rank"]).copy()
    pool["relevance"] = scale_scores(pool)

    rows = []
    for user_id, user_pool in pool.groupby("user_id", sort=True):
        app_ids = user_pool["app_id"].to_numpy()
        vectors = tag_vectors(app_ids, tfidf, row_of_app)
        similarity = vectors @ vectors.T
        picked = mmr_select(user_pool["relevance"].to_numpy(), similarity, lam, FINAL_K)
        for new_rank, position in enumerate(picked, start=1):
            rows.append({"user_id": user_id, "app_id": app_ids[position], "rank": new_rank})
    return pd.DataFrame(rows, columns=["user_id", "app_id", "rank"])


def intra_list_diversity(recs: pd.DataFrame, tfidf: np.ndarray, row_of_app: pd.Series, k: int = FINAL_K) -> float:
    """Average over users of the mean tag dissimilarity (1 - cosine) between every pair in their top k."""
    top = recs[recs["rank"] <= k]
    per_user = []
    for _, user_top in top.groupby("user_id"):
        n = len(user_top)
        if n < 2:
            continue
        vectors = tag_vectors(user_top["app_id"].to_numpy(), tfidf, row_of_app)
        similarity = vectors @ vectors.T
        # Sum over all pairs, minus the diagonal (each game with itself), divided by the number of pairs.
        pair_similarity = (similarity.sum() - np.trace(similarity)) / (n * (n - 1))
        per_user.append(1 - pair_similarity)
    return float(np.mean(per_user))


def score_setting(recs, targets, like_counts, n_catalog, tfidf, row_of_app) -> dict[str, float]:
    """All the numbers we compare for one re-ranked top 10."""
    all_targets = ranking_metrics(recs, targets, recall_ks=[FINAL_K], ndcg_k=FINAL_K)
    new_targets = ranking_metrics(recs, targets[targets["is_new_game"]], recall_ks=[FINAL_K], ndcg_k=FINAL_K)
    beyond = beyond_accuracy(recs, like_counts, n_catalog, k=FINAL_K)
    return {
        "recall@10": all_targets["recall@10"],
        "ndcg@10": all_targets["ndcg@10"],
        "new_recall@10": new_targets["recall@10"],
        "coverage@10": beyond["coverage@10"],
        "head_share@10": beyond["head_share@10"],
        "diversity@10": intra_list_diversity(recs, tfidf, row_of_app),
    }


def choose_setting(results: dict[str, dict[str, float]], baseline_ndcg: float, objective: str, lower_is_better: bool) -> str:
    """Name of the setting with the best `objective` among those keeping NDCG@10 >= (1 - MAX_NDCG_LOSS) x baseline."""
    min_ndcg = (1 - MAX_NDCG_LOSS) * baseline_ndcg
    best_name, best_value = None, None
    for name, metrics in results.items():
        if metrics["ndcg@10"] < min_ndcg:
            continue
        value = metrics[objective]
        if lower_is_better:
            value = -value
        if best_value is None or value > best_value:
            best_name, best_value = name, value
    return best_name


def rerank(
    processed_dir: Path = PROCESSED_DIR,
    split_dir: Path = SPLIT_DIR,
    ranked_path: Path = RANKED_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Try every penalty strength and MMR strength on the held-out val users and save the comparison."""
    ranked = pd.read_parquet(ranked_path)
    catalog = pd.read_parquet(processed_dir / "catalog.parquet", columns=["app_id", "tags"])
    targets = pd.read_parquet(split_dir / "val_targets.parquet")
    targets = targets[targets["user_id"].isin(ranked["user_id"].unique())]
    reviews = pd.read_parquet(processed_dir / "reviews.parquet", columns=["app_id", "date", "is_recommended"])
    likes = reviews[reviews["is_recommended"] & (reviews["date"] < VAL_CUTOFF)]
    like_counts = likes["app_id"].value_counts()
    del reviews, likes

    tfidf = build_tag_matrix(catalog["tags"])
    row_of_app = pd.Series(np.arange(len(catalog)), index=catalog["app_id"])
    max_log_likes = ranked["log_likes"].max()
    n_catalog = len(catalog)

    penalty_results = {}
    for alpha in ALPHAS:
        recs = popularity_penalty(ranked, alpha, max_log_likes)
        penalty_results[f"alpha={alpha}"] = score_setting(recs, targets, like_counts, n_catalog, tfidf, row_of_app)
        print(f"alpha={alpha}: {penalty_results[f'alpha={alpha}']}")

    mmr_results = {}
    for lam in LAMBDAS:
        recs = mmr_rerank(ranked, tfidf, row_of_app, lam)
        mmr_results[f"lambda={lam}"] = score_setting(recs, targets, like_counts, n_catalog, tfidf, row_of_app)
        print(f"lambda={lam}: {mmr_results[f'lambda={lam}']}")

    baseline_ndcg = penalty_results["alpha=0.0"]["ndcg@10"]
    results = {
        "popularity_penalty": penalty_results,
        "mmr": mmr_results,
        "chosen": {
            "popularity_penalty": choose_setting(penalty_results, baseline_ndcg, "head_share@10", lower_is_better=True),
            "mmr": choose_setting(mmr_results, baseline_ndcg, "diversity@10", lower_is_better=False),
        },
    }
    print("chosen:", results["chosen"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    rerank()
