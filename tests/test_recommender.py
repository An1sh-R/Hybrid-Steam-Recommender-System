"""Tests for steam_rec.recommender with the tiny `toy_artifacts` fixture (conftest.py) and its fake model."""

import pandas as pd
import pytest

from steam_rec.recommender import recommend, score_and_rerank, search_games


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
