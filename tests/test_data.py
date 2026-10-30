"""Tests for steam_rec.data using the tiny raw files from the `raw_dir` fixture (conftest.py)."""

import pandas as pd

from steam_rec.data import flag_non_games, load_games, load_metadata, load_reviews


def test_flag_non_games_matches_whole_words_only():
    titles = pd.Series([
        "Rain World - Soundtrack",
        "Persona 5 OST",
        "Little Nightmares The Hideaway DLC",
        "Ghost of Tsushima",
        "Lost Ark",
        "Hollow Knight",
    ])
    flags = flag_non_games(titles)
    assert flags.tolist() == [True, True, True, False, False, False]


def test_load_games_drops_steam_deck_and_flags_non_games(raw_dir):
    games = load_games(raw_dir)

    assert "steam_deck" not in games.columns
    assert games["is_non_game"].tolist() == [False, True]
    assert pd.api.types.is_datetime64_any_dtype(games["date_release"])


def test_load_metadata_reads_tags_as_lists(raw_dir):
    meta = load_metadata(raw_dir)

    assert meta.loc[0, "tags"] == ["Metroidvania", "Souls-like"]
    assert meta.loc[1, "tags"] == []


def test_load_reviews_keeps_used_columns_with_small_dtypes(raw_dir):
    reviews = load_reviews(raw_dir)

    assert sorted(reviews.columns) == sorted(["user_id", "app_id", "date", "is_recommended", "hours"])
    assert reviews["user_id"].dtype == "int32"
    assert reviews["app_id"].dtype == "int32"
    assert reviews["is_recommended"].dtype == "bool"
    assert reviews["hours"].dtype == "float32"


def test_load_reviews_keeps_latest_duplicate_review(raw_dir):
    reviews = load_reviews(raw_dir)

    user_1_game_10 = reviews[(reviews["user_id"] == 1) & (reviews["app_id"] == 10)]
    assert len(user_1_game_10) == 1
    assert user_1_game_10["is_recommended"].item() is True
    assert len(reviews) == 2
