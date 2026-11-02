"""Tests for steam_rec.recommender with the tiny `toy_artifacts` fixture (conftest.py) and its fake model."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from steam_rec.recommender import load_artifacts, recommend, score_and_rerank, search_games, top_games


def test_search_is_case_insensitive_skips_non_games_and_puts_popular_first(toy_artifacts):
    found = search_games("hollow", toy_artifacts.games)
    assert found["title"].tolist() == ["Hollow Knight"]

    found = search_games("E", toy_artifacts.games)
    assert found["title"].tolist() == ["Hades", "Dead Cells", "Celeste"]


def top_titles(result):
    """The titles of the top recommendations, in order."""
    return [rec["title"] for rec in result["recommendations"]]


def test_recommend_never_returns_seeds_or_non_games(toy_artifacts):
    recs = recommend([1, 2], toy_artifacts)["recommendations"]
    app_ids = [rec["app_id"] for rec in recs]

    assert 1 not in app_ids and 2 not in app_ids
    assert 13 not in app_ids
    assert sorted(app_ids) == [10, 11, 12]


def test_recommend_orders_by_ranker_score_with_alpha_zero(toy_artifacts):
    result = recommend([1, 2], toy_artifacts, alpha=0.0)

    # Fake scores: Dead Cells 0.05 + 1.3 = 1.35, Ori 0.9, Celeste 0.10.
    assert top_titles(result) == ["Dead Cells", "Ori", "Celeste"]


def test_explanations_come_from_the_retriever_that_ranked_the_game_higher(toy_artifacts):
    recs = {rec["title"]: rec["reason"] for rec in recommend([1, 2], toy_artifacts)["recommendations"]}

    # Celeste was only found by co-occurrence.
    assert recs["Celeste"]["type"] == "players_also_liked"
    assert recs["Celeste"]["text"] == "Players who liked Hollow Knight also liked this"

    # Ori was only found by content; shared tags keep Ori's own tag order.
    assert recs["Ori"]["type"] == "similar_tags"
    assert recs["Ori"]["text"] == "Because you liked Hollow Knight: shares Metroidvania, Platformer"

    # Dead Cells: content rank 1 beats co-occurrence rank 3 (the soundtrack ranked first there before
    # being filtered out). Its best content seed is Hades (0.7 > 0.6).
    assert recs["Dead Cells"]["type"] == "similar_tags"
    assert recs["Dead Cells"]["seed_app_id"] == 2
    assert recs["Dead Cells"]["shared_tags"] == ["Action Roguelike"]


def test_recommend_respects_n_and_rejects_unknown_games(toy_artifacts):
    assert len(recommend([1, 2], toy_artifacts, n=2)["recommendations"]) == 2

    with pytest.raises(ValueError):
        recommend([1, 999], toy_artifacts)


def test_rows_skip_games_already_in_the_top_n(toy_artifacts):
    # Only 3 real candidates, and all of them make the top 10, so the rows have nothing left.
    result = recommend([1, 2], toy_artifacts)
    assert result["similar_tags"] == []
    assert result["players_also_liked"] == []

    # With a top 1 (Dead Cells), each row gets the game only its retriever found.
    result = recommend([1, 2], toy_artifacts, n=1, alpha=0.0)
    assert top_titles(result) == ["Dead Cells"]
    assert [rec["title"] for rec in result["similar_tags"]] == ["Ori"]
    assert [rec["title"] for rec in result["players_also_liked"]] == ["Celeste"]


def test_rows_use_their_own_retriever_order_reason_and_skip_each_other(toy_artifacts):
    # No top list at all, so the rows are filled straight from the two retrievers.
    result = recommend([1, 2], toy_artifacts, n=0)
    similar = result["similar_tags"]
    also_liked = result["players_also_liked"]

    # Content order: Dead Cells (content rank 1), then Ori.
    assert [rec["title"] for rec in similar] == ["Dead Cells", "Ori"]
    for rec in similar:
        assert rec["reason"]["type"] == "similar_tags"

    # Dead Cells was found by both retrievers but is already in the row above, so it's skipped here;
    # the soundtrack is never shown.
    assert [rec["title"] for rec in also_liked] == ["Celeste"]
    assert also_liked[0]["reason"]["type"] == "players_also_liked"


def test_row_size_limits_each_row(toy_artifacts):
    result = recommend([1, 2], toy_artifacts, n=0, row_size=1)
    assert [rec["title"] for rec in result["similar_tags"]] == ["Dead Cells"]
    assert [rec["title"] for rec in result["players_also_liked"]] == ["Celeste"]


def test_score_and_rerank_ranks_every_candidate_then_penalises_to_top_10(toy_artifacts):
    # 2 users x 15 candidates; the fake model scores cooc_score + content_score.
    rows = []
    for user in [1, 2]:
        for app in range(15):
            rows.append({"user_id": user, "app_id": app, "cooc_score": app / 10, "content_score": 0.0,
                         "cooc_rank": 1, "content_rank": 201, "log_likes": 0.0, "like_share": 0.5,
                         "price": 0.0, "age_years": 1.0})
    candidates = pd.DataFrame(rows)
    ranked, final = score_and_rerank(candidates, toy_artifacts.model, max_log_likes=12.0)

    assert ranked.groupby("user_id").size().tolist() == [15, 15]
    assert final.groupby("user_id").size().tolist() == [10, 10]
    # No popularity differences here, so the penalty keeps the model's order: app 14 first.
    assert final[final["user_id"] == 1]["app_id"].tolist()[:3] == [14, 13, 12]


def test_load_artifacts_keeps_only_the_needed_columns_with_small_ids(tmp_path):
    serve_dir = tmp_path / "serve"
    content_dir = tmp_path / "content"
    serve_dir.mkdir()
    content_dir.mkdir()

    pd.DataFrame({"app_id": [1], "title": ["Hades"]}).to_parquet(serve_dir / "games.parquet")
    pd.DataFrame({"app_id": [1], "log_likes": [2.5]}).to_parquet(serve_dir / "game_features.parquet")
    # The real files have extra columns that scoring never uses.
    pd.DataFrame({"app_id": [1], "neighbor_id": [2], "both_liked": [7], "similarity": [0.5],
                  "also_liked_share": [0.1]}).to_parquet(serve_dir / "cooc_neighbors.parquet")
    pd.DataFrame({"app_id": np.array([1], dtype="int64"), "neighbor_id": np.array([2], dtype="int64"),
                  "similarity": np.array([0.5], dtype="float32")}).to_parquet(content_dir / "neighbors.parquet")
    # A real (tiny) LightGBM model file, since load_artifacts reads one from disk.
    data = lgb.Dataset(np.array([[0.0], [1.0], [2.0], [3.0]]), label=[0.0, 1.0, 2.0, 3.0])
    model = lgb.train({"objective": "regression", "min_data_in_leaf": 1, "verbose": -1}, data, num_boost_round=1)
    model.save_model(str(tmp_path / "ranker.txt"))

    artifacts = load_artifacts(serve_dir, content_dir, tmp_path / "ranker.txt")

    assert artifacts.cooc_neighbors.columns.tolist() == ["app_id", "neighbor_id", "similarity"]
    assert artifacts.content_neighbors["app_id"].dtype == "int32"
    assert artifacts.content_neighbors["neighbor_id"].dtype == "int32"
    assert artifacts.max_log_likes == 2.5


def test_top_games_are_the_most_liked_real_games(toy_artifacts):
    found = top_games(toy_artifacts.games, toy_artifacts.game_features, n=3)
    # Toy log_likes: Hollow Knight 12, Hades 11, Dead Cells 10, Celeste 9, Ori 8, soundtrack 2.
    assert found["title"].tolist() == ["Hollow Knight", "Hades", "Dead Cells"]

    # The soundtrack (a non-game) never appears, however many are asked for.
    everything = top_games(toy_artifacts.games, toy_artifacts.game_features, n=100)
    assert 13 not in everything["app_id"].tolist()
    assert len(everything) == 5


def test_top_games_skip_software(toy_artifacts):
    # Make Hollow Knight look like Wallpaper Engine: Steam tags software "Software".
    games = toy_artifacts.games.copy()
    games["tags"] = games["tags"].astype(object)
    games.at[0, "tags"] = ["Utilities", "Software"]

    found = top_games(games, toy_artifacts.game_features, n=2)
    assert found["title"].tolist() == ["Hades", "Dead Cells"]
