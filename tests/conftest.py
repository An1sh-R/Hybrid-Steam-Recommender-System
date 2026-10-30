"""Shared pytest fixtures: tiny hand-written versions of the raw Kaggle files."""

import pytest

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
