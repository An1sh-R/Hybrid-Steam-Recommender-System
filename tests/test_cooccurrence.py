"""Tests for steam_rec.cooccurrence on tiny hand-built likes."""

import math

import pandas as pd
import pytest

from steam_rec.cooccurrence import build_like_matrix, likes_before, recommend, top_neighbors

CUTOFF = pd.Timestamp("2022-01-01")

# Three players: games A=1, B=2, C=3.
#   user 10 likes A, B      user 20 likes A, B, C      user 30 likes A, C
# Likes per game: A=3, B=2, C=2.  Pairs liked together: A-B=2, A-C=2, B-C=1.
TOY_LIKES = pd.DataFrame({
    "user_id": [10, 10, 20, 20, 20, 30, 30],
    "app_id": [1, 2, 1, 2, 3, 1, 3],
})


def make_reviews(rows):
    """Build a reviews table from (user_id, app_id, date, is_recommended) tuples."""
    reviews = pd.DataFrame(rows, columns=["user_id", "app_id", "date", "is_recommended"])
    reviews["date"] = pd.to_datetime(reviews["date"])
    return reviews


def neighbor_row(neighbors, app_id, neighbor_id):
    """Return the single row for the pair (app_id -> neighbor_id)."""
    match = neighbors[(neighbors["app_id"] == app_id) & (neighbors["neighbor_id"] == neighbor_id)]
    assert len(match) == 1
    return match.iloc[0]


def test_likes_before_respects_cutoff_dislikes_and_single_likes():
    reviews = make_reviews([
        (1, 101, "2021-01-01", True),
        (1, 102, "2021-02-01", True),
        (1, 103, "2021-03-01", False),   # dislike -> ignored
        (1, 104, "2022-06-01", True),    # after cutoff -> ignored
        (2, 101, "2021-01-01", True),    # user 2 has a single like -> dropped
    ])
    likes = likes_before(reviews, CUTOFF)

    assert likes["user_id"].tolist() == [1, 1]
    assert sorted(likes["app_id"].tolist()) == [101, 102]


def test_likes_before_keeps_most_recent_likes_per_user():
    reviews = make_reviews([
        (1, 101, "2021-01-01", True),
        (1, 102, "2021-02-01", True),
        (1, 103, "2021-03-01", True),
    ])
    likes = likes_before(reviews, CUTOFF, max_per_user=2)

    assert sorted(likes["app_id"].tolist()) == [102, 103]


def test_build_like_matrix_marks_each_like():
    matrix, app_ids = build_like_matrix(TOY_LIKES)

    assert matrix.shape == (3, 3)
    assert matrix.sum() == 7
    assert sorted(app_ids.tolist()) == [1, 2, 3]


def test_top_neighbors_cosine_and_share_match_hand_calculation():
    matrix, app_ids = build_like_matrix(TOY_LIKES)
    neighbors = top_neighbors(matrix, app_ids, top_k=10, min_both_liked=1)

    a_to_b = neighbor_row(neighbors, 1, 2)
    assert a_to_b["both_liked"] == 2
    assert a_to_b["similarity"] == pytest.approx(2 / math.sqrt(3 * 2))
    assert a_to_b["also_liked_share"] == pytest.approx(2 / 3)

    b_to_a = neighbor_row(neighbors, 2, 1)
    assert b_to_a["similarity"] == pytest.approx(a_to_b["similarity"])
    assert b_to_a["also_liked_share"] == pytest.approx(1.0)

    b_to_c = neighbor_row(neighbors, 2, 3)
    assert b_to_c["similarity"] == pytest.approx(1 / math.sqrt(2 * 2))


def test_top_neighbors_never_pairs_a_game_with_itself():
    matrix, app_ids = build_like_matrix(TOY_LIKES)
    neighbors = top_neighbors(matrix, app_ids, top_k=10, min_both_liked=1)

    assert (neighbors["app_id"] != neighbors["neighbor_id"]).all()


def test_top_neighbors_applies_min_both_liked_and_top_k():
    matrix, app_ids = build_like_matrix(TOY_LIKES)

    # B-C was liked together by only 1 user, so min_both_liked=2 removes it.
    filtered = top_neighbors(matrix, app_ids, top_k=10, min_both_liked=2)
    b_neighbors = filtered[filtered["app_id"] == 2]["neighbor_id"].tolist()
    assert b_neighbors == [1]

    # top_k=1 keeps only B's best neighbour: A (0.82) beats C (0.50).
    best_only = top_neighbors(matrix, app_ids, top_k=1, min_both_liked=1)
    assert best_only[best_only["app_id"] == 2]["neighbor_id"].tolist() == [1]


def test_chunk_size_does_not_change_the_result():
    matrix, app_ids = build_like_matrix(TOY_LIKES)
    one_at_a_time = top_neighbors(matrix, app_ids, top_k=10, min_both_liked=1, chunk_size=1)
    all_at_once = top_neighbors(matrix, app_ids, top_k=10, min_both_liked=1, chunk_size=100)

    pd.testing.assert_frame_equal(one_at_a_time, all_at_once)


# A hand-made neighbour table for testing recommend().
#   seed 1 -> 2 (0.9), 3 (0.5), 4 (0.2)        seed 2 -> 1 (0.9), 3 (0.3), 5 (0.4)
TOY_NEIGHBORS = pd.DataFrame({
    "app_id": [1, 1, 1, 2, 2, 2],
    "neighbor_id": [2, 3, 4, 1, 3, 5],
    "both_liked": [9, 5, 2, 9, 3, 4],
    "similarity": [0.9, 0.5, 0.2, 0.9, 0.3, 0.4],
    "also_liked_share": [0.8, 0.4, 0.1, 0.7, 0.6, 0.3],
})


def test_recommend_sums_similarity_over_seeds_and_skips_seeds():
    seeds = pd.DataFrame({"user_id": [7, 7], "app_id": [1, 2]})
    recs = recommend(seeds, TOY_NEIGHBORS)

    # Game 3: 0.5 (from seed 1) + 0.3 (from seed 2) = 0.8. Games 1 and 2 are seeds, so skipped.
    assert recs["app_id"].tolist() == [3, 5, 4]
    assert recs["score"].tolist() == pytest.approx([0.8, 0.4, 0.2])
    assert recs["rank"].tolist() == [1, 2, 3]


def test_recommend_explains_with_the_strongest_seed():
    seeds = pd.DataFrame({"user_id": [7, 7], "app_id": [1, 2]})
    recs = recommend(seeds, TOY_NEIGHBORS)
    game_3 = recs[recs["app_id"] == 3].iloc[0]

    # Seed 1 gives game 3 more similarity (0.5) than seed 2 does (0.3).
    assert game_3["best_seed"] == 1
    assert game_3["best_seed_share"] == pytest.approx(0.4)


def test_recommend_handles_each_user_separately_and_respects_top_n():
    seeds = pd.DataFrame({"user_id": [7, 7, 8], "app_id": [1, 2, 2]})
    recs = recommend(seeds, TOY_NEIGHBORS, top_n=2)

    # User 8 only gave game 2, so game 1 is a fair recommendation for them.
    assert recs[recs["user_id"] == 8]["app_id"].tolist() == [1, 5]
    assert recs[recs["user_id"] == 7]["app_id"].tolist() == [3, 5]


def test_recommend_returns_nothing_for_seeds_without_neighbours():
    seeds = pd.DataFrame({"user_id": [9], "app_id": [999]})
    recs = recommend(seeds, TOY_NEIGHBORS)

    assert len(recs) == 0
