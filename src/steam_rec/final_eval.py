"""The one-time final evaluation on the untouched test period (decisions D4, D22, D33).

Everything was designed, trained and tuned on the val period. This scores the finished system,
unchanged, on the 50k test users: seeds before TEST_CUTOFF (2022-08-13), targets after it.
- retrievers fitted before TEST_CUTOFF (co-occurrence test fit; content limited to games released
  before the data ends)
- game features as of TEST_CUTOFF
- the val-trained ranker (models/ranker.txt), no retraining
- the popularity penalty with the chosen alpha = 0.3

Methods compared on the same users: popularity, co-occurrence alone, content alone, the ranker,
and the final system (ranker + penalty). Each is scored on its top 10; Recall@200 shows the
candidate pool for the methods that produce 200.

Output: metrics/test.json. Run with: uv run python -m steam_rec.final_eval
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from steam_rec.candidates import released_before
from steam_rec.content import CONTENT_DIR, build_tag_matrix
from steam_rec.cooccurrence import COOCCURRENCE_DIR
from steam_rec.data import PROJECT_ROOT
from steam_rec.evaluate import popularity_recommend, ranking_metrics
from steam_rec.prepare import PROCESSED_DIR
from steam_rec.ranker import FEATURES, MODEL_PATH, build_candidates, game_features, pool_recall, retriever_only
from steam_rec.recommender import ALPHA, score_and_rerank
from steam_rec.rerank import score_setting
from steam_rec.serve_build import FULL_CUTOFF
from steam_rec.split import SPLIT_DIR, TEST_CUTOFF

METRICS_PATH = PROJECT_ROOT / "metrics" / "test.json"


def final_evaluation(
    processed_dir: Path = PROCESSED_DIR,
    split_dir: Path = SPLIT_DIR,
    cooccurrence_dir: Path = COOCCURRENCE_DIR,
    content_dir: Path = CONTENT_DIR,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Score every method once on the test users and write metrics/test.json."""
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    catalog_columns = ["app_id", "price_final", "date_release", "tags"]
    catalog = pd.read_parquet(processed_dir / "catalog.parquet", columns=catalog_columns)
    seeds = pd.read_parquet(split_dir / "test_seeds.parquet")
    targets = pd.read_parquet(split_dir / "test_targets.parquet")
    cooc_neighbors = pd.read_parquet(cooccurrence_dir / "test_neighbors.parquet")
    content_neighbors = pd.read_parquet(content_dir / "neighbors.parquet")

    # The test period runs to the end of the data, so content may suggest anything released by then.
    content_neighbors = released_before(content_neighbors, catalog, FULL_CUTOFF)

    features = game_features(reviews, catalog, TEST_CUTOFF)
    likes = reviews[reviews["is_recommended"] & (reviews["date"] < TEST_CUTOFF)]
    like_counts = likes["app_id"].value_counts()
    del reviews, likes  # the biggest table; not needed any more

    candidates = build_candidates(seeds, cooc_neighbors, content_neighbors)
    candidates = candidates.merge(features, on="app_id", how="left")

    model = lgb.Booster(model_file=str(model_path))
    max_log_likes = float(features["log_likes"].max())
    ranked, final = score_and_rerank(candidates[["user_id", "app_id"] + FEATURES], model, max_log_likes, ALPHA)

    methods = {
        "popularity": popularity_recommend(like_counts, seeds),
        "cooccurrence": retriever_only(candidates, "cooc"),
        "content": retriever_only(candidates, "content"),
        "ranker": ranked,
        "final": final,
    }

    tfidf = build_tag_matrix(catalog["tags"])
    row_of_app = pd.Series(np.arange(len(catalog)), index=catalog["app_id"])

    results = {"users": int(targets["user_id"].nunique()), "recall@pool": pool_recall(candidates, targets)}
    for name, recs in methods.items():
        metrics = score_setting(recs, targets, like_counts, len(catalog), tfidf, row_of_app)
        if name != "final":
            metrics["recall@200"] = ranking_metrics(recs, targets, recall_ks=[200], ndcg_k=10)["recall@200"]
        results[name] = metrics
        print(f"{name}: {metrics}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    final_evaluation()
