"""Tests for steam_rec.rerank on tiny hand-built lists and tag vectors."""

import numpy as np
import pandas as pd
import pytest

from steam_rec.rerank import (
    choose_setting,
    intra_list_diversity,
    mmr_select,
    popularity_penalty,
    scale_scores,
)

# Three tag "directions": games 1 and 2 are near-duplicates, game 3 is completely different.
TFIDF = np.array([
    [1.0, 0.0],
    [0.99, 0.141],
    [0.0, 1.0],
])
ROW_OF_APP = pd.Series([0, 1, 2], index=[1, 2, 3])


def test_scale_scores_per_user_and_constant_scores():
    recs = pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "score": [10.0, 5.0, 0.0, 3.0, 3.0]})
    assert scale_scores(recs).tolist() == pytest.approx([1.0, 0.5, 0.0, 1.0, 1.0])


def test_popularity_penalty_zero_keeps_order_and_strong_penalty_demotes_blockbuster():
    recs = pd.DataFrame({
        "user_id": [1, 1],
        "app_id": [100, 200],
        "score": [2.0, 1.0],
        "log_likes": [12.0, 3.0],
    })
    unchanged = popularity_penalty(recs, alpha=0.0, max_log_likes=12.0)
    assert unchanged["app_id"].tolist() == [100, 200]

    # Game 100: relevance 1 - 1.0 x (12/12) = 0.  Game 200: 0 - 1.0 x (3/12) = -0.25. Still 100 first.
    # With alpha = 2: 100 -> -1.0, 200 -> -0.5, so the less popular game moves up.
    penalised = popularity_penalty(recs, alpha=2.0, max_log_likes=12.0)
    assert penalised["app_id"].tolist() == [200, 100]


def test_mmr_with_lambda_one_is_pure_relevance():
    similarity = TFIDF @ TFIDF.T
    picked = mmr_select(np.array([1.0, 0.9, 0.5]), similarity, lam=1.0, k=3)
    assert picked == [0, 1, 2]


def test_mmr_skips_a_near_duplicate():
    similarity = TFIDF @ TFIDF.T
    # Second pick: game 2 scores 0.5 x 0.9 - 0.5 x 0.99 < 0; game 3 scores 0.5 x 0.5 - 0.5 x 0 > 0.
    picked = mmr_select(np.array([1.0, 0.9, 0.5]), similarity, lam=0.5, k=2)
    assert picked == [0, 2]


def test_intra_list_diversity():
    near_duplicates = pd.DataFrame({"user_id": [1, 1], "app_id": [1, 2], "rank": [1, 2]})
    opposites = pd.DataFrame({"user_id": [1, 1], "app_id": [1, 3], "rank": [1, 2]})

    assert intra_list_diversity(near_duplicates, TFIDF, ROW_OF_APP) == pytest.approx(0.01, abs=0.001)
    assert intra_list_diversity(opposites, TFIDF, ROW_OF_APP) == pytest.approx(1.0)


def test_choose_setting_respects_the_ndcg_budget():
    results = {
        "a": {"ndcg@10": 0.100, "head_share@10": 0.75},
        "b": {"ndcg@10": 0.097, "head_share@10": 0.60},
        "c": {"ndcg@10": 0.080, "head_share@10": 0.30},
    }
    # c has the lowest head share but loses 20% NDCG; b loses only 3%, so b wins.
    assert choose_setting(results, baseline_ndcg=0.100, objective="head_share@10", lower_is_better=True) == "b"
