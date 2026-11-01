"""Tests for steam_rec.evaluate with metrics small enough to work out by hand."""

import math

import pandas as pd
import pytest

from steam_rec.evaluate import beyond_accuracy, evaluate_recommendations, popularity_recommend, ranking_metrics

# User 1 liked games 1 and 2 later; user 2 liked game 4 later.
TARGETS = pd.DataFrame({
    "user_id": [1, 1, 2],
    "app_id": [1, 2, 4],
    "is_new_game": [False, True, False],
})

# User 1 gets games 1, 3, 2 at ranks 1, 2, 3. User 2 gets nothing.
RECS = pd.DataFrame({
    "user_id": [1, 1, 1],
    "app_id": [1, 3, 2],
    "rank": [1, 2, 3],
})


def test_recall_counts_users_without_recommendations_as_zero():
    metrics = ranking_metrics(RECS, TARGETS, recall_ks=[1, 3], ndcg_k=3)

    # User 1: 1 of 2 targets in the top 1, 2 of 2 in the top 3. User 2: nothing found.
    assert metrics["recall@1"] == pytest.approx((0.5 + 0.0) / 2)
    assert metrics["recall@3"] == pytest.approx((1.0 + 0.0) / 2)


def test_ndcg_matches_hand_calculation():
    metrics = ranking_metrics(RECS, TARGETS, recall_ks=[3], ndcg_k=3)

    # User 1 hits at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.5.
    # Best possible: ranks 1 and 2: IDCG = 1/log2(2) + 1/log2(3).
    user_1 = 1.5 / (1 + 1 / math.log2(3))
    assert metrics["ndcg@3"] == pytest.approx((user_1 + 0.0) / 2)


def test_ndcg_is_one_for_a_perfect_ranking():
    perfect = pd.DataFrame({"user_id": [1, 1], "app_id": [1, 2], "rank": [1, 2]})
    metrics = ranking_metrics(perfect, TARGETS[TARGETS["user_id"] == 1], recall_ks=[2], ndcg_k=3)

    assert metrics["ndcg@3"] == pytest.approx(1.0)


def test_evaluate_recommendations_splits_warm_and_new_targets():
    results = evaluate_recommendations(RECS, TARGETS)

    # Warm targets: user 1's game 1 (found at rank 1) and user 2's game 4 (not found).
    assert results["warm"]["recall@10"] == pytest.approx(0.5)
    # New targets: only user 1's game 2, found at rank 3.
    assert results["new"]["recall@10"] == pytest.approx(1.0)


def test_popularity_recommend_gives_most_liked_games_minus_seeds():
    like_counts = pd.Series([50, 40, 30, 20], index=[100, 200, 300, 400])
    seeds = pd.DataFrame({"user_id": [1, 2], "app_id": [100, 300]})
    recs = popularity_recommend(like_counts, seeds, top_n=2)

    assert recs[recs["user_id"] == 1]["app_id"].tolist() == [200, 300]
    assert recs[recs["user_id"] == 2]["app_id"].tolist() == [100, 200]


def test_beyond_accuracy_coverage_and_head_share():
    recs = pd.DataFrame({"user_id": [1, 1, 2], "app_id": [100, 200, 100], "rank": [1, 2, 1]})
    like_counts = pd.Series(range(200, 0, -1), index=range(100, 300))
    metrics = beyond_accuracy(recs, like_counts, n_catalog=10, k=10)

    # 2 distinct games out of a 10-game catalog. Head = top 1% of 200 games = game 100 and 101.
    assert metrics["coverage@10"] == pytest.approx(0.2)
    assert metrics["head_share@10"] == pytest.approx(2 / 3)
