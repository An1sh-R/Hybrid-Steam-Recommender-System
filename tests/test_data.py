"""Tests for steam_rec.data using tiny hand-written versions of the raw files."""

import pandas as pd

from steam_rec.data import flag_non_games, load_games, load_metadata, load_reviews

GAMES_CSV = """app_id,title,date_release,win,mac,linux,rating,positive_ratio,user_reviews,price_final,price_original,discount,steam_deck
10,Hollow Knight,2017-02-24,true,true,true,Overwhelmingly Positive,97,300000,14.99,14.99,0.0,true
20,Hollow Knight - Soundtrack,2017-02-24,true,true,true,Positive,95,50,4.99,4.99,0.0,true
"""

METADATA_JSON = """{"app_id":10,"description":"Forge your own path.","tags":["Metroidvania","Souls-like"]}
{"app_id":20,"description":"","tags":[]}
"""

REVIEWS_CSV = """app_id,helpful,funny,date,is_recommended,hours,user_id,review_id
10,0,0,2020-01-01,false,1.5,1,0
10,3,0,2021-06-01,true,40.0,1,1
20,0,0,2020-05-05,true,2.0,2,2
"""


def write_raw_files(folder):
    """Write the three tiny raw files into `folder`."""
    (folder / "games.csv").write_text(GAMES_CSV)
    (folder / "games_metadata.json").write_text(METADATA_JSON)
    (folder / "recommendations.csv").write_text(REVIEWS_CSV)


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


def test_load_games_drops_steam_deck_and_flags_non_games(tmp_path):
    write_raw_files(tmp_path)
    games = load_games(tmp_path)

    assert "steam_deck" not in games.columns
    assert games["is_non_game"].tolist() == [False, True]
    assert pd.api.types.is_datetime64_any_dtype(games["date_release"])


def test_load_metadata_reads_tags_as_lists(tmp_path):
    write_raw_files(tmp_path)
    meta = load_metadata(tmp_path)

    assert meta.loc[0, "tags"] == ["Metroidvania", "Souls-like"]
    assert meta.loc[1, "tags"] == []


def test_load_reviews_keeps_used_columns_with_small_dtypes(tmp_path):
    write_raw_files(tmp_path)
    reviews = load_reviews(tmp_path)

    assert sorted(reviews.columns) == sorted(["user_id", "app_id", "date", "is_recommended", "hours"])
    assert reviews["user_id"].dtype == "int32"
    assert reviews["app_id"].dtype == "int32"
    assert reviews["is_recommended"].dtype == "bool"
    assert reviews["hours"].dtype == "float32"


def test_load_reviews_keeps_latest_duplicate_review(tmp_path):
    write_raw_files(tmp_path)
    reviews = load_reviews(tmp_path)

    user_1_game_10 = reviews[(reviews["user_id"] == 1) & (reviews["app_id"] == 10)]
    assert len(user_1_game_10) == 1
    assert user_1_game_10["is_recommended"].item() is True
    assert len(reviews) == 2
