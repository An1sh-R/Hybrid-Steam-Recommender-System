"""Tests for steam_rec.candidates."""

import pandas as pd
import pytest

from steam_rec.candidates import keep_top_n, recommend_from_neighbors, released_before, remove_seeds

# A hand-made neighbour table, as any retriever would produce.
#   seed 1 -> 2 (0.9), 3 (0.5), 4 (0.2)        seed 2 -> 1 (0.9), 3 (0.3), 5 (0.4)
TOY_NEIGHBORS = pd.DataFrame({
    "app_id": [1, 1, 1, 2, 2, 2],
    "neighbor_id": [2, 3, 4, 1, 3, 5],
    "similarity": [0.9, 0.5, 0.2, 0.9, 0.3, 0.4],
})


def test_remove_seeds_only_drops_that_users_seeds():
    candidates = pd.DataFrame({"user_id": [1, 1, 2], "app_id": [10, 20, 10], "score": [0.5, 0.4, 0.3]})
    seeds = pd.DataFrame({"user_id": [1], "app_id": [10]})
    kept = remove_seeds(candidates, seeds)

    # Game 10 is user 1's seed, but still a valid candidate for user 2.
    assert list(zip(kept["user_id"], kept["app_id"])) == [(1, 20), (2, 10)]


def test_keep_top_n_sorts_by_score_breaks_ties_by_app_id_and_ranks():
    candidates = pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "app_id": [30, 10, 20, 40],
        "score": [0.2, 0.5, 0.5, 0.9],
    })
    ranked = keep_top_n(candidates, top_n=2)

    assert ranked[ranked["user_id"] == 1]["app_id"].tolist() == [10, 20]
    assert ranked[ranked["user_id"] == 1]["rank"].tolist() == [1, 2]
    assert ranked[ranked["user_id"] == 2]["rank"].tolist() == [1]


def test_recommend_sums_similarity_over_seeds_and_skips_seeds():
    seeds = pd.DataFrame({"user_id": [7, 7], "app_id": [1, 2]})
    recs = recommend_from_neighbors(seeds, TOY_NEIGHBORS)

    # Game 3: 0.5 (from seed 1) + 0.3 (from seed 2) = 0.8. Games 1 and 2 are seeds, so skipped.
    assert recs["app_id"].tolist() == [3, 5, 4]
    assert recs["score"].tolist() == pytest.approx([0.8, 0.4, 0.2])
    assert recs["rank"].tolist() == [1, 2, 3]


def test_recommend_explains_with_the_strongest_seed():
    seeds = pd.DataFrame({"user_id": [7, 7], "app_id": [1, 2]})
    recs = recommend_from_neighbors(seeds, TOY_NEIGHBORS)
    game_3 = recs[recs["app_id"] == 3].iloc[0]

    # Seed 1 gives game 3 more similarity (0.5) than seed 2 does (0.3).
    assert game_3["best_seed"] == 1


def test_recommend_picks_the_lower_seed_id_when_seeds_tie():
    neighbors = pd.DataFrame({"app_id": [1, 2], "neighbor_id": [3, 3], "similarity": [0.5, 0.5]})
    seeds = pd.DataFrame({"user_id": [7, 7], "app_id": [2, 1]})
    recs = recommend_from_neighbors(seeds, neighbors)

    assert recs["best_seed"].tolist() == [1]


def test_recommend_handles_each_user_separately_and_respects_top_n():
    seeds = pd.DataFrame({"user_id": [7, 7, 8], "app_id": [1, 2, 2]})
    recs = recommend_from_neighbors(seeds, TOY_NEIGHBORS, top_n=2)

    # User 8 only gave game 2, so game 1 is a fair recommendation for them.
    assert recs[recs["user_id"] == 8]["app_id"].tolist() == [1, 5]
    assert recs[recs["user_id"] == 7]["app_id"].tolist() == [3, 5]


def test_recommend_returns_nothing_for_seeds_without_neighbours():
    seeds = pd.DataFrame({"user_id": [9], "app_id": [999]})
    recs = recommend_from_neighbors(seeds, TOY_NEIGHBORS)

    assert len(recs) == 0


def test_released_before_drops_games_that_did_not_exist_yet():
    neighbors = pd.DataFrame({"app_id": [1, 1], "neighbor_id": [10, 20], "similarity": [0.9, 0.8]})
    catalog = pd.DataFrame({"app_id": [10, 20], "date_release": pd.to_datetime(["2021-05-01", "2023-03-01"])})
    kept = released_before(neighbors, catalog, pd.Timestamp("2022-08-13"))

    assert kept["neighbor_id"].tolist() == [10]
