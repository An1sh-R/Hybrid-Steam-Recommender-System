"""Shared pytest fixtures: tiny raw Kaggle files, and tiny serving artifacts with a fake model."""

import pandas as pd
import pytest

from steam_rec.recommender import Artifacts

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


@pytest.fixture
def raw_dir(tmp_path):
    """A temporary folder holding the three tiny raw files."""
    (tmp_path / "games.csv").write_text(GAMES_CSV)
    (tmp_path / "games_metadata.json").write_text(METADATA_JSON)
    (tmp_path / "recommendations.csv").write_text(REVIEWS_CSV)
    return tmp_path


# --- Tiny serving artifacts, shared by test_recommender.py and test_api.py ---
# Seeds: 1 (Hollow Knight) and 2 (Hades). Candidates: 10, 11, 12; 13 is a soundtrack.
TOY_GAMES = pd.DataFrame({
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
TOY_COOC = pd.DataFrame({
    "app_id": [1, 1, 1],
    "neighbor_id": [10, 11, 13],
    "similarity": [0.10, 0.05, 0.20],
})

# Content: seed 1 -> Ori (12) and Dead Cells (11); seed 2 -> Dead Cells (11).
TOY_CONTENT = pd.DataFrame({
    "app_id": [1, 1, 2],
    "neighbor_id": [12, 11, 11],
    "similarity": [0.9, 0.6, 0.7],
})

TOY_FEATURES = pd.DataFrame({
    "app_id": [1, 2, 10, 11, 12, 13],
    "log_likes": [12.0, 11.0, 9.0, 10.0, 8.0, 2.0],
    "like_share": [0.95] * 6,
    "price": [14.99, 24.99, 19.99, 24.99, 19.99, 4.99],
    "age_years": [5.0, 2.0, 4.0, 4.0, 2.0, 5.0],
})


class FakeModel:
    """Stands in for the LightGBM booster: score = cooc_score + content_score."""

    def predict(self, features):
        """Score each candidate row the same way every time."""
        return (features["cooc_score"] + features["content_score"]).to_numpy()


def make_toy_artifacts():
    """Artifacts built from the toy tables above."""
    return Artifacts(
        games=TOY_GAMES,
        cooc_neighbors=TOY_COOC,
        content_neighbors=TOY_CONTENT,
        game_features=TOY_FEATURES,
        model=FakeModel(),
        max_log_likes=12.0,
    )


@pytest.fixture
def toy_artifacts():
    """A fresh set of tiny serving artifacts."""
    return make_toy_artifacts()
