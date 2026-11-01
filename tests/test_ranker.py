"""Tests for steam_rec.ranker on tiny hand-built tables."""

import numpy as np
import pandas as pd
import pytest

from steam_rec.ranker import (
    FEATURES,
    MISSING_RANK,
    add_labels,
    build_candidates,
    game_features,
    pool_recall,
    rank_with_model,
    sample_negatives,
    split_users,
    train_ranker,
)

CUTOFF = pd.Timestamp("2022-01-01")


def test_build_candidates_unions_both_retrievers_and_fills_gaps():
    seeds = pd.DataFrame({"user_id": [1], "app_id": [10]})
    cooc_neighbors = pd.DataFrame({"app_id": [10, 10], "neighbor_id": [20, 30], "similarity": [0.9, 0.5]})
    content_neighbors = pd.DataFrame({"app_id": [10, 10], "neighbor_id": [30, 40], "similarity": [0.8, 0.7]})
    candidates = build_candidates(seeds, cooc_neighbors, content_neighbors).set_index("app_id")

    # 20 only from co-occurrence, 40 only from content, 30 from both.
    assert sorted(candidates.index.tolist()) == [20, 30, 40]
    assert candidates.loc[30, "cooc_score"] == pytest.approx(0.5)
    assert candidates.loc[30, "content_score"] == pytest.approx(0.8)
    assert candidates.loc[20, "content_score"] == 0.0
    assert candidates.loc[20, "content_rank"] == MISSING_RANK
    assert candidates.loc[40, "cooc_rank"] == MISSING_RANK


def test_game_features_only_use_reviews_before_the_cutoff():
    reviews = pd.DataFrame({
        "app_id": [1, 1, 1, 1],
        "date": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01", "2022-06-01"]),
        "is_recommended": [True, True, False, True],
    })
    catalog = pd.DataFrame({
        "app_id": [1, 2],
        "price_final": [9.99, 0.0],
        "date_release": pd.to_datetime(["2020-01-01", "2023-01-01"]),
    })
    features = game_features(reviews, catalog, CUTOFF).set_index("app_id")

    # Game 1: 2 likes out of 3 reviews before the cutoff (the 2022 review is ignored).
    assert features.loc[1, "log_likes"] == pytest.approx(np.log1p(2))
    assert features.loc[1, "like_share"] == pytest.approx((2 + 1) / (3 + 2))
    # Game 2: no reviews yet, released a year after the cutoff.
    assert features.loc[2, "log_likes"] == 0.0
    assert features.loc[2, "like_share"] == pytest.approx(0.5)
    assert features.loc[2, "age_years"] == pytest.approx(-1.0, abs=0.01)
    assert features.loc[2, "price"] == 0.0


def test_add_labels_marks_targets():
    candidates = pd.DataFrame({"user_id": [1, 1, 2], "app_id": [10, 20, 10]})
    targets = pd.DataFrame({"user_id": [1], "app_id": [20], "is_new_game": [False]})
    labelled = add_labels(candidates, targets)

    assert labelled["label"].tolist() == [0, 1, 0]


def test_split_users_is_disjoint_sized_and_reproducible():
    user_ids = np.arange(100)
    train_a, eval_a = split_users(user_ids, eval_share=0.2, random_seed=1)
    train_b, eval_b = split_users(user_ids, eval_share=0.2, random_seed=1)

    assert len(eval_a) == 20 and len(train_a) == 80
    assert set(train_a).isdisjoint(set(eval_a))
    assert eval_a.tolist() == eval_b.tolist()


def test_sample_negatives_keeps_positives_caps_negatives_drops_users_without_positives():
    table = pd.DataFrame({
        "user_id": [1] * 6 + [2] * 3,
        "app_id": [10, 11, 12, 13, 14, 15, 10, 11, 12],
        "label": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    sampled = sample_negatives(table, n_negatives=2)

    assert sampled["user_id"].unique().tolist() == [1]
    assert sampled["label"].sum() == 1
    assert (sampled["label"] == 0).sum() == 2


def test_pool_recall_averages_per_user():
    candidates = pd.DataFrame({"user_id": [1, 1, 2], "app_id": [10, 20, 30]})
    targets = pd.DataFrame({"user_id": [1, 1, 2], "app_id": [10, 99, 30]})

    # User 1: 1 of 2 targets in the pool; user 2: 1 of 1.
    assert pool_recall(candidates, targets) == pytest.approx((0.5 + 1.0) / 2)


def test_ranker_learns_to_put_positives_first():
    # 40 users, 10 candidates each; the liked game is always the one with the highest content_score,
    # while cooc_score is pure noise. A working ranker must learn to use content_score.
    rng = np.random.default_rng(0)
    rows = []
    for user in range(40):
        liked = rng.integers(10)
        for app in range(10):
            rows.append({
                "user_id": user,
                "app_id": app,
                "cooc_score": rng.random(),
                "cooc_rank": rng.integers(1, 201),
                "content_score": 1.0 if app == liked else rng.random() * 0.5,
                "content_rank": 1 if app == liked else rng.integers(2, 201),
                "log_likes": rng.random(),
                "like_share": rng.random(),
                "price": rng.random() * 20,
                "age_years": rng.random() * 5,
                "label": int(app == liked),
            })
    table = pd.DataFrame(rows)
    params = {"objective": "lambdarank", "n_estimators": 30, "min_child_samples": 5, "verbose": -1, "random_state": 0}
    model = train_ranker(table, params)
    ranked = rank_with_model(model, table[["user_id", "app_id"] + FEATURES], top_n=1)

    top_choices = ranked.merge(table[["user_id", "app_id", "label"]], on=["user_id", "app_id"])
    assert top_choices["label"].mean() == 1.0
