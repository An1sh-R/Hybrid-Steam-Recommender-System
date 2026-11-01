"""Tests for steam_rec.recommender with the tiny `toy_artifacts` fixture (conftest.py) and its fake model."""

import pytest

from steam_rec.recommender import recommend, search_games


def test_search_is_case_insensitive_skips_non_games_and_puts_popular_first(toy_artifacts):
    found = search_games("hollow", toy_artifacts.games)
    assert found["title"].tolist() == ["Hollow Knight"]

    found = search_games("E", toy_artifacts.games)
    assert found["title"].tolist() == ["Hades", "Dead Cells", "Celeste"]


def test_recommend_never_returns_seeds_or_non_games(toy_artifacts):
    recs = recommend([1, 2], toy_artifacts)
    app_ids = [rec["app_id"] for rec in recs]

    assert 1 not in app_ids and 2 not in app_ids
    assert 13 not in app_ids
    assert sorted(app_ids) == [10, 11, 12]


def test_recommend_orders_by_ranker_score_with_alpha_zero(toy_artifacts):
    recs = recommend([1, 2], toy_artifacts, alpha=0.0)

    # Fake scores: Dead Cells 0.05 + 1.3 = 1.35, Ori 0.9, Celeste 0.10.
    assert [rec["title"] for rec in recs] == ["Dead Cells", "Ori", "Celeste"]


def test_explanations_come_from_the_retriever_that_ranked_the_game_higher(toy_artifacts):
    recs = {rec["title"]: rec["reason"] for rec in recommend([1, 2], toy_artifacts)}

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
    assert len(recommend([1, 2], toy_artifacts, n=2)) == 2

    with pytest.raises(ValueError):
        recommend([1, 999], toy_artifacts)
