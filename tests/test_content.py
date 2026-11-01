"""Tests for steam_rec.content on a handful of made-up games."""

import numpy as np
import pandas as pd
import pytest

from steam_rec.content import build_tag_matrix, top_content_neighbors

# Games 1 and 2 have identical tags. Game 3 shares only the common tag "Indie" with them.
# Game 4 is a soundtrack (not allowed as a neighbour). Game 5 has no tags at all.
TAGS = pd.Series([
    ["Indie", "Metroidvania", "Souls-like"],
    ["Indie", "Metroidvania", "Souls-like"],
    ["Indie", "Racing"],
    ["Indie", "Metroidvania", "Souls-like"],
    [],
])
APP_IDS = np.array([1, 2, 3, 4, 5])
CAN_BE_NEIGHBOR = np.array([True, True, True, False, True])


def neighbors_of(neighbors, app_id):
    """Return the neighbour ids of one game, best first."""
    return neighbors[neighbors["app_id"] == app_id]["neighbor_id"].tolist()


def test_tag_matrix_rows_have_length_one_or_zero():
    tfidf = build_tag_matrix(TAGS)
    row_lengths = np.linalg.norm(tfidf, axis=1)

    assert row_lengths[:4] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert row_lengths[4] == 0.0


def test_rare_tags_weigh_more_than_common_ones():
    tfidf = build_tag_matrix(TAGS)
    # Columns are tags in alphabetical order: Indie, Metroidvania, Racing, Souls-like.
    indie_weight, racing_weight = tfidf[2, 0], tfidf[2, 2]

    # "Indie" is on 4 of 5 games, "Racing" on 1, so Racing must count more for game 3.
    assert racing_weight > indie_weight


def test_tags_stored_as_numpy_arrays_work_too():
    # After a parquet round trip, tags come back as numpy arrays rather than lists.
    as_arrays = pd.Series([np.array(tags, dtype=object) for tags in TAGS])
    np.testing.assert_allclose(build_tag_matrix(as_arrays), build_tag_matrix(TAGS))


def test_identical_tags_are_the_best_neighbour_with_similarity_one():
    neighbors = top_content_neighbors(build_tag_matrix(TAGS), APP_IDS, CAN_BE_NEIGHBOR)
    best_for_1 = neighbors[neighbors["app_id"] == 1].iloc[0]

    assert best_for_1["neighbor_id"] == 2
    assert best_for_1["similarity"] == pytest.approx(1.0)


def test_never_itself_never_a_non_game_never_zero_similarity():
    neighbors = top_content_neighbors(build_tag_matrix(TAGS), APP_IDS, CAN_BE_NEIGHBOR)

    assert (neighbors["app_id"] != neighbors["neighbor_id"]).all()
    assert 4 not in neighbors["neighbor_id"].tolist()
    assert (neighbors["similarity"] > 0).all()
    # Game 5 has no tags, so it neither has neighbours nor is anyone's neighbour.
    assert 5 not in neighbors["app_id"].tolist()
    assert 5 not in neighbors["neighbor_id"].tolist()


def test_the_soundtrack_still_gets_neighbours_itself():
    # A user could type a soundtrack as a seed; it should still lead to real games.
    neighbors = top_content_neighbors(build_tag_matrix(TAGS), APP_IDS, CAN_BE_NEIGHBOR)

    assert neighbors_of(neighbors, 4) == [1, 2, 3]


def test_top_k_and_chunk_size():
    tfidf = build_tag_matrix(TAGS)
    best_only = top_content_neighbors(tfidf, APP_IDS, CAN_BE_NEIGHBOR, top_k=1)
    assert neighbors_of(best_only, 3) == [1]

    one_at_a_time = top_content_neighbors(tfidf, APP_IDS, CAN_BE_NEIGHBOR, chunk_size=1)
    all_at_once = top_content_neighbors(tfidf, APP_IDS, CAN_BE_NEIGHBOR, chunk_size=100)
    pd.testing.assert_frame_equal(one_at_a_time, all_at_once)
