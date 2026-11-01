"""Tests for steam_rec.recommender with tiny hand-made artifacts and a fake model."""

import pandas as pd
import pytest

from steam_rec.recommender import Artifacts, recommend, search_games

# Seeds: 1 (Hollow Knight-like) and 2. Candidates: 10, 11, 12; 13 is a soundtrack.
GAMES = pd.DataFrame({
    "app_id": [1, 2, 10, 11, 12, 13],
    "title": ["Hollow Knight", "Hades", "Celeste", "Dead Cells", "Ori", "Hollow Knight - Soundtrack"],
    "release_year": [2017, 2020, 2018, 2018, 2020, 2017],
    "tags": [
        ["Metroidvania", "Souls-like", "Platformer"],
        ["Action Roguelike", "Hack and Slash"],
        ["Platformer", "Difficult"],
        ["Action Roguelike", "Metroidvania", "Souls-like", "Platformer"],
        ["Metroidvania", "Platformer"],
        ["Soundtrack"],
    ],
    "is_non_game": [False, False, False, False, False, True],
    "price_final": [14.99, 24.99, 19.99, 24.99, 19.99, 4.99],
    "positive_ratio": [97, 98, 95, 93, 95, 90],
    "user_reviews": [300000, 200000, 80000, 150000, 60000, 500],
})

# Co-occurrence: seed 1 -> Celeste (10), Dead Cells (11), the soundtrack (13).
COOC = pd.DataFrame({
    "app_id": [1, 1, 1],
    "neighbor_id": [10, 11, 13],
    "similarity": [0.10, 0.05, 0.20],
})

# Content: seed 1 -> Ori (12) and Dead Cells (11); seed 2 -> Dead Cells (11).
CONTENT = pd.DataFrame({
    "app_id": [1, 1, 2],
    "neighbor_id": [12, 11, 11],
    "similarity": [0.9, 0.6, 0.7],
})

FEATURES = pd.DataFrame({
    "app_id": [1, 2, 10, 11, 12, 13],
    "log_likes": [12.0, 11.0, 9.0, 10.0, 8.0, 2.0],
    "like_share": [0.95] * 6,
    "price": [14.99, 24.99, 19.99, 24.99, 19.99, 4.99],
    "age_years": [5.0, 2.0, 4.0, 4.0, 2.0, 5.0],
})


class FakeModel:
    """Stands in for the LightGBM booster: score = cooc_score + content_score."""

    def predict(self, features):
        return (features["cooc_score"] + features["content_score"]).to_numpy()


def make_artifacts():
    """Artifacts built from the tables above."""
    return Artifacts(
        games=GAMES,
        cooc_neighbors=COOC,
        content_neighbors=CONTENT,
        game_features=FEATURES,
        model=FakeModel(),
        max_log_likes=12.0,
    )


def test_search_is_case_insensitive_skips_non_games_and_puts_popular_first():
    found = search_games("hollow", GAMES)
    assert found["title"].tolist() == ["Hollow Knight"]

    found = search_games("E", GAMES)
    assert found["title"].tolist() == ["Hades", "Dead Cells", "Celeste"]


def test_recommend_never_returns_seeds_or_non_games():
    recs = recommend([1, 2], make_artifacts())
    app_ids = [rec["app_id"] for rec in recs]

    assert 1 not in app_ids and 2 not in app_ids
    assert 13 not in app_ids
    assert sorted(app_ids) == [10, 11, 12]


def test_recommend_orders_by_ranker_score_with_alpha_zero():
    recs = recommend([1, 2], make_artifacts(), alpha=0.0)

    # Fake scores: Dead Cells 0.05 + 1.3 = 1.35, Ori 0.9, Celeste 0.10.
    assert [rec["title"] for rec in recs] == ["Dead Cells", "Ori", "Celeste"]


def test_explanations_come_from_the_retriever_that_ranked_the_game_higher():
    recs = {rec["title"]: rec["reason"] for rec in recommend([1, 2], make_artifacts())}

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


def test_recommend_respects_n_and_rejects_unknown_games():
    assert len(recommend([1, 2], make_artifacts(), n=2)) == 2

    with pytest.raises(ValueError):
        recommend([1, 999], make_artifacts())
