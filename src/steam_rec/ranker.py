"""LightGBM ranker: re-orders the merged candidates of both retrievers (decisions D2, D25).

Training data comes from the val period only (the test period stays untouched):
1. candidates = union of each user's top 200 from co-occurrence and top 200 from content
2. features per (user, candidate): both retrievers' scores and ranks, plus game facts known
   before the cutoff (popularity, like share, price, age)
3. label = 1 if the user liked that game in the val period, else 0
4. val users are split 80 / 20: the ranker learns on 80% and is scored on the other 20%
   against co-occurrence alone and content alone, on exactly the same users

LambdaRank: gradient-boosted trees trained to put each user's positives above their negatives,
directly optimising NDCG.

Outputs: models/ranker.txt (the LightGBM model) and metrics/ranker_val.json.
Run with: uv run python -m steam_rec.ranker
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from steam_rec.candidates import TOP_N, keep_top_n, recommend_from_neighbors
from steam_rec.content import CONTENT_DIR
from steam_rec.cooccurrence import COOCCURRENCE_DIR
from steam_rec.data import PROJECT_ROOT
from steam_rec.evaluate import beyond_accuracy, evaluate_recommendations
from steam_rec.prepare import PROCESSED_DIR
from steam_rec.split import RANDOM_SEED, SPLIT_DIR, TEST_CUTOFF, VAL_CUTOFF

MODEL_PATH = PROJECT_ROOT / "models" / "ranker.txt"
METRICS_PATH = PROJECT_ROOT / "metrics" / "ranker_val.json"

FEATURES = [
    "cooc_score",
    "cooc_rank",
    "content_score",
    "content_rank",
    "log_likes",
    "like_share",
    "price",
    "age_years",
]
MISSING_RANK = TOP_N + 1
EVAL_USER_SHARE = 0.2
NEGATIVES_PER_USER = 50

RANKER_PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "random_state": RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def build_candidates(seeds: pd.DataFrame, cooc_neighbors: pd.DataFrame, content_neighbors: pd.DataFrame) -> pd.DataFrame:
    """Union of both retrievers' top candidates per user, with each retriever's score and rank."""
    cooc = recommend_from_neighbors(seeds, cooc_neighbors)
    cooc = cooc.rename(columns={"score": "cooc_score", "rank": "cooc_rank", "best_seed": "cooc_best_seed"})
    content = recommend_from_neighbors(seeds, content_neighbors)
    content = content.rename(columns={"score": "content_score", "rank": "content_rank", "best_seed": "content_best_seed"})

    # Outer merge: keep a candidate if either retriever found it.
    candidates = cooc.merge(content, on=["user_id", "app_id"], how="outer")

    # A retriever that didn't find the candidate gives it score 0 and a rank just past its list.
    candidates["cooc_score"] = candidates["cooc_score"].fillna(0.0)
    candidates["content_score"] = candidates["content_score"].fillna(0.0)
    candidates["cooc_rank"] = candidates["cooc_rank"].fillna(MISSING_RANK)
    candidates["content_rank"] = candidates["content_rank"].fillna(MISSING_RANK)
    return candidates


def game_features(reviews: pd.DataFrame, catalog: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per-game facts known at `cutoff`: log popularity, smoothed like share, price, age in years."""
    before = reviews[reviews["date"] < cutoff]
    per_game = before.groupby("app_id").agg(likes=("is_recommended", "sum"), reviews=("is_recommended", "size"))

    features = catalog[["app_id", "price_final", "date_release"]].merge(per_game, on="app_id", how="left")
    features["likes"] = features["likes"].fillna(0)
    features["reviews"] = features["reviews"].fillna(0)

    features["log_likes"] = np.log1p(features["likes"])
    # Smoothed share of positive reviews: a game with no reviews gets 0.5 instead of dividing by zero.
    features["like_share"] = (features["likes"] + 1) / (features["reviews"] + 2)
    features["price"] = features["price_final"]
    # Negative for games released after the cutoff.
    features["age_years"] = (cutoff - features["date_release"]).dt.days / 365.25
    return features[["app_id", "log_likes", "like_share", "price", "age_years"]]


def add_labels(candidates: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Add `label` = 1 where the candidate is one of the user's targets, else 0."""
    marked = targets[["user_id", "app_id"]].copy()
    marked["label"] = 1
    labelled = candidates.merge(marked, on=["user_id", "app_id"], how="left")
    labelled["label"] = labelled["label"].fillna(0).astype(int)
    return labelled


def split_users(
    user_ids: np.ndarray, eval_share: float = EVAL_USER_SHARE, random_seed: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split users into (train_users, eval_users), reproducibly."""
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(np.sort(user_ids))
    n_eval = int(len(shuffled) * eval_share)
    return np.sort(shuffled[n_eval:]), np.sort(shuffled[:n_eval])


def sample_negatives(table: pd.DataFrame, n_negatives: int = NEGATIVES_PER_USER, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Keep every positive and up to `n_negatives` random negatives per user; drop users with no positive."""
    positives_per_user = table.groupby("user_id")["label"].sum()
    users_with_positive = positives_per_user[positives_per_user > 0].index
    table = table[table["user_id"].isin(users_with_positive)]

    positives = table[table["label"] == 1]
    negatives = table[table["label"] == 0]
    # Shuffle the negatives, then take the first n of each user.
    negatives = negatives.sample(frac=1.0, random_state=random_seed)
    negatives = negatives.groupby("user_id").head(n_negatives)

    sampled = pd.concat([positives, negatives])
    return sampled.sort_values(["user_id", "app_id"]).reset_index(drop=True)


def train_ranker(table: pd.DataFrame, params: dict | None = None) -> lgb.LGBMRanker:
    """Fit a LambdaRank model; each user's rows form one group to rank."""
    if params is None:
        params = RANKER_PARAMS
    # LightGBM needs each group's rows next to each other, plus the size of every group in order.
    table = table.sort_values("user_id")
    group_sizes = table.groupby("user_id", sort=True).size().to_numpy()

    model = lgb.LGBMRanker(**params)
    model.fit(table[FEATURES], table["label"], group=group_sizes)
    return model


def rank_with_model(model: lgb.LGBMRanker, candidates: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Score every candidate with the model and keep each user's top `top_n`."""
    scored = candidates[["user_id", "app_id"]].copy()
    scored["score"] = model.predict(candidates[FEATURES])
    return keep_top_n(scored, top_n)


def retriever_only(candidates: pd.DataFrame, retriever: str) -> pd.DataFrame:
    """One retriever's own ranking (cooc or content) restricted to the same candidate table."""
    found = candidates[candidates[f"{retriever}_rank"] <= TOP_N]
    ranked = found[["user_id", "app_id"]].copy()
    ranked["score"] = found[f"{retriever}_score"]
    return keep_top_n(ranked)


def pool_recall(candidates: pd.DataFrame, targets: pd.DataFrame) -> float:
    """Share of each user's targets that are anywhere in their candidate pool, averaged over users."""
    matched = targets.merge(candidates[["user_id", "app_id"]], on=["user_id", "app_id"], how="left", indicator=True)
    in_pool = matched["_merge"] == "both"
    return float(in_pool.groupby(matched["user_id"]).mean().mean())


def ranker(
    processed_dir: Path = PROCESSED_DIR,
    split_dir: Path = SPLIT_DIR,
    cooccurrence_dir: Path = COOCCURRENCE_DIR,
    content_dir: Path = CONTENT_DIR,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Train the ranker on 80% of val users, score it on the other 20%, save model and metrics."""
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    catalog = pd.read_parquet(processed_dir / "catalog.parquet", columns=["app_id", "price_final", "date_release"])
    seeds = pd.read_parquet(split_dir / "val_seeds.parquet")
    targets = pd.read_parquet(split_dir / "val_targets.parquet")
    cooc_neighbors = pd.read_parquet(cooccurrence_dir / "val_neighbors.parquet")
    content_neighbors = pd.read_parquet(content_dir / "neighbors.parquet")

    # Same rule as evaluate.py: content may only suggest games that exist by the end of the val period.
    released_ids = catalog.loc[catalog["date_release"] < TEST_CUTOFF, "app_id"]
    content_neighbors = content_neighbors[content_neighbors["neighbor_id"].isin(released_ids)]

    features_by_game = game_features(reviews, catalog, VAL_CUTOFF)
    likes = reviews[reviews["is_recommended"] & (reviews["date"] < VAL_CUTOFF)]
    like_counts = likes["app_id"].value_counts()
    del reviews, likes  # the biggest table; not needed any more

    candidates = build_candidates(seeds, cooc_neighbors, content_neighbors)
    candidates = candidates.merge(features_by_game, on="app_id", how="left")
    candidates = add_labels(candidates, targets)

    train_users, eval_users = split_users(seeds["user_id"].unique())
    train_table = sample_negatives(candidates[candidates["user_id"].isin(train_users)])
    print(f"training on {train_table['user_id'].nunique():,} users, {len(train_table):,} rows")
    model = train_ranker(train_table)

    eval_candidates = candidates[candidates["user_id"].isin(eval_users)]
    eval_targets = targets[targets["user_id"].isin(eval_users)]
    methods = {
        "cooccurrence": retriever_only(eval_candidates, "cooc"),
        "content": retriever_only(eval_candidates, "content"),
        "ranker": rank_with_model(model, eval_candidates),
    }

    results = {"pool": {"eval_users": len(eval_users), "recall@pool": pool_recall(eval_candidates, eval_targets)}}
    for name, recs in methods.items():
        method_results = evaluate_recommendations(recs, eval_targets)
        method_results["beyond"] = beyond_accuracy(recs, like_counts, len(catalog))
        results[name] = method_results
        print(f"{name:13s} recall@10 {method_results['all']['recall@10']:.4f}  ndcg@10 {method_results['all']['ndcg@10']:.4f}")

    gains = model.booster_.feature_importance(importance_type="gain")
    importance = {}
    for feature, gain in zip(FEATURES, gains):
        importance[feature] = float(gain / gains.sum())
    results["feature_importance"] = importance

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ranker()
