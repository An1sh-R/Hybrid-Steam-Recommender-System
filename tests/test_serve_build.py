"""Tests for steam_rec.serve_build on a tiny processed dataset."""

import pandas as pd

from steam_rec.serve_build import GAME_COLUMNS, serve_build, serving_games


def make_catalog():
    """Three games, as catalog.parquet would hold them."""
    return pd.DataFrame({
        "app_id": [1, 2, 3],
        "title": ["A", "B", "C"],
        "date_release": pd.to_datetime(["2015-03-01", "2020-07-15", "2022-11-30"]),
        "tags": [["RPG"], ["RPG", "Indie"], ["Indie"]],
        "is_non_game": [False, False, False],
        "price_final": [9.99, 4.99, 0.0],
        "positive_ratio": [90, 80, 70],
        "user_reviews": [1000, 500, 20],
        "description": ["", "", ""],
    })


def test_serving_games_keeps_display_columns_and_release_year():
    games = serving_games(make_catalog())

    assert list(games.columns) == GAME_COLUMNS
    assert games["release_year"].tolist() == [2015, 2020, 2022]


def test_serve_build_uses_reviews_up_to_the_last_day(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    make_catalog().to_parquet(processed_dir / "catalog.parquet", index=False)

    # Five players who each liked games 1 and 2 (enough for MIN_BOTH_LIKED = 5),
    # the last of them on the very last day of the data.
    rows = []
    for user in range(5):
        rows.append({"user_id": user, "app_id": 1, "date": "2022-01-01", "is_recommended": True, "hours": 5.0})
        rows.append({"user_id": user, "app_id": 2, "date": "2022-12-31", "is_recommended": True, "hours": 5.0})
    reviews = pd.DataFrame(rows)
    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews.to_parquet(processed_dir / "reviews.parquet", index=False)

    out_dir = tmp_path / "serve"
    serve_build(processed_dir, out_dir)

    neighbors = pd.read_parquet(out_dir / "cooc_neighbors.parquet")
    features = pd.read_parquet(out_dir / "game_features.parquet").set_index("app_id")
    assert sorted(zip(neighbors["app_id"], neighbors["neighbor_id"], strict=True)) == [(1, 2), (2, 1)]
    # Game 2's likes on 2022-12-31 count, and game 3 (no reviews) still gets features.
    assert features.loc[2, "log_likes"] > 0
    assert features.loc[3, "log_likes"] == 0
    assert (out_dir / "games.parquet").exists()
