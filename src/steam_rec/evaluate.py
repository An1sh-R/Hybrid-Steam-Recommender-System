"""Offline evaluation of the retrievers on the validation set (decisions D4, D22).

For every method, each val user gets up to 200 ranked games from their seeds, scored against the
games they actually liked later (targets):
- Recall@K: share of a user's targets found in their top K (K = 10, 50, 200)
- NDCG@10: like Recall@10, but a hit at rank 1 counts more than a hit at rank 10
Both are averaged over users and reported for all / warm / new-game targets (D10).
Plus, for the top 10: catalog coverage and the share of recommendations that are blockbusters.

Methods: a popularity baseline (most-liked games, same list for everyone), co-occurrence and
content (tag similarity). Content may only recommend games released before the val period ends.
The test set is only scored once, at the very end of the project.

Output: metrics/retrieval_val.json (a DVC metrics file: `uv run dvc metrics show`).
Run with: uv run python -m steam_rec.evaluate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from steam_rec.candidates import TOP_N, keep_top_n, recommend_from_neighbors, released_before, remove_seeds
from steam_rec.content import CONTENT_DIR
from steam_rec.cooccurrence import COOCCURRENCE_DIR
from steam_rec.data import PROJECT_ROOT
from steam_rec.prepare import PROCESSED_DIR
from steam_rec.split import MAX_SEEDS, SPLIT_DIR, TEST_CUTOFF, VAL_CUTOFF

METRICS_PATH = PROJECT_ROOT / "metrics" / "retrieval_val.json"

RECALL_KS = [10, 50, 200]
NDCG_K = 10
BEYOND_K = 10
HEAD_SHARE = 0.01


def popularity_recommend(like_counts: pd.Series, seeds: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Give every user the most-liked games (`like_counts`, sorted high to low), minus their seeds."""
    # Take a few extra games, since up to MAX_SEEDS of them may be removed as seeds.
    popular = like_counts.head(top_n + MAX_SEEDS)
    popular_games = pd.DataFrame({"app_id": popular.index, "score": popular.to_numpy().astype(float)})

    users = pd.DataFrame({"user_id": seeds["user_id"].unique()})
    candidates = users.merge(popular_games, how="cross")
    candidates = remove_seeds(candidates, seeds)
    return keep_top_n(candidates, top_n)


def ranking_metrics(
    recs: pd.DataFrame,
    targets: pd.DataFrame,
    recall_ks: list[int] = RECALL_KS,
    ndcg_k: int = NDCG_K,
) -> dict[str, float]:
    """Recall@K for each K and NDCG@ndcg_k, averaged over every user in `targets`."""
    # One row per target, with the rank it got (NaN if it wasn't recommended at all).
    hits = targets.merge(recs[["user_id", "app_id", "rank"]], on=["user_id", "app_id"], how="left")
    rank = hits["rank"].fillna(np.inf)
    targets_per_user = targets.groupby("user_id").size()

    metrics = {}
    for k in recall_ks:
        found = rank <= k
        found_per_user = found.groupby(hits["user_id"]).sum()
        recall_per_user = found_per_user / targets_per_user
        metrics[f"recall@{k}"] = float(recall_per_user.mean())

    # DCG: each hit in the top ndcg_k is worth 1 / log2(rank + 1), so rank 1 = 1.0, rank 3 = 0.5.
    gain = np.where(rank <= ndcg_k, 1 / np.log2(rank + 1), 0.0)
    dcg_per_user = pd.Series(gain).groupby(hits["user_id"]).sum()

    # Ideal DCG: the user's targets placed at ranks 1, 2, 3, ... (at most ndcg_k of them).
    ideal_gains = 1 / np.log2(np.arange(2, ndcg_k + 2))
    best_possible_hits = targets_per_user.clip(upper=ndcg_k)
    ideal_dcg_per_user = best_possible_hits.map(lambda n: ideal_gains[:n].sum())

    ndcg_per_user = dcg_per_user / ideal_dcg_per_user
    metrics[f"ndcg@{ndcg_k}"] = float(ndcg_per_user.mean())
    return metrics


def evaluate_recommendations(recs: pd.DataFrame, targets: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Ranking metrics for all targets, warm targets only, and new-game targets only."""
    target_groups = {
        "all": targets,
        "warm": targets[~targets["is_new_game"]],
        "new": targets[targets["is_new_game"]],
    }
    results = {}
    for group, group_targets in target_groups.items():
        results[group] = ranking_metrics(recs, group_targets)
    return results


def beyond_accuracy(recs: pd.DataFrame, like_counts: pd.Series, n_catalog: int, k: int = BEYOND_K) -> dict[str, float]:
    """Coverage of the catalog and share of blockbuster games in everyone's top k."""
    top = recs[recs["rank"] <= k]
    n_head = max(1, int(len(like_counts) * HEAD_SHARE))
    head_games = like_counts.head(n_head).index
    return {
        f"coverage@{k}": top["app_id"].nunique() / n_catalog,
        f"head_share@{k}": float(top["app_id"].isin(head_games).mean()),
    }


def evaluate(
    processed_dir: Path = PROCESSED_DIR,
    split_dir: Path = SPLIT_DIR,
    cooccurrence_dir: Path = COOCCURRENCE_DIR,
    content_dir: Path = CONTENT_DIR,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Score every method on the validation set and write the metrics JSON."""
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    catalog = pd.read_parquet(processed_dir / "catalog.parquet", columns=["app_id", "date_release"])
    seeds = pd.read_parquet(split_dir / "val_seeds.parquet")
    targets = pd.read_parquet(split_dir / "val_targets.parquet")
    cooccurrence_neighbors = pd.read_parquet(cooccurrence_dir / "val_neighbors.parquet")
    content_neighbors = pd.read_parquet(content_dir / "neighbors.parquet")

    # Popularity = number of likes before the val cutoff, most-liked first.
    likes = reviews[reviews["is_recommended"] & (reviews["date"] < VAL_CUTOFF)]
    like_counts = likes["app_id"].value_counts()

    # Content knows every catalog game; only allow games that exist by the end of the val period.
    content_neighbors = released_before(content_neighbors, catalog, TEST_CUTOFF)

    methods = {
        "popularity": popularity_recommend(like_counts, seeds),
        "cooccurrence": recommend_from_neighbors(seeds, cooccurrence_neighbors),
        "content": recommend_from_neighbors(seeds, content_neighbors),
    }
    n_catalog = len(catalog)

    results = {}
    for name, recs in methods.items():
        method_results = evaluate_recommendations(recs, targets)
        method_results["beyond"] = beyond_accuracy(recs, like_counts, n_catalog)
        results[name] = method_results

        all_targets = method_results["all"]
        print(
            f"{name:13s} recall@10 {all_targets['recall@10']:.4f}  recall@200 {all_targets['recall@200']:.4f}  "
            f"ndcg@10 {all_targets['ndcg@10']:.4f}"
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate()
