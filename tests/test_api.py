"""Tests for steam_rec.api: real HTTP requests to the app, running on the tiny toy artifacts."""

import pytest
from fastapi.testclient import TestClient

from steam_rec.api import create_app


@pytest.fixture
def client(toy_artifacts):
    """A test client for an app that loads the toy artifacts instead of the real files."""

    def load_toy_artifacts():
        """Hand the app the fixture's tiny artifacts."""
        return toy_artifacts

    app = create_app(load=load_toy_artifacts)
    # `with` runs the app's startup (loading the artifacts) and shutdown around the test.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_matching_games(client):
    response = client.get("/search", params={"q": "hollow"})

    assert response.status_code == 200
    assert response.json() == [{"app_id": 1, "title": "Hollow Knight", "release_year": 2017}]


def test_search_needs_at_least_two_characters(client):
    assert client.get("/search", params={"q": "h"}).status_code == 422


def test_recommend_returns_games_with_reasons(client):
    response = client.post("/recommend", json={"app_ids": [1, 2, 10]})

    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    titles = [rec["title"] for rec in recommendations]
    assert sorted(titles) == ["Dead Cells", "Ori"]
    assert recommendations[0]["reason"]["text"].startswith("Because you liked")
    # Every candidate already made the top list, so both extra rows are empty (but present).
    assert response.json()["similar_tags"] == []
    assert response.json()["players_also_liked"] == []


@pytest.mark.parametrize("app_ids", [
    [1, 2],               # too few
    [1, 2, 10, 11, 12, 13],  # too many
    [1, 1, 2],            # same game twice
])
def test_recommend_validates_the_input(client, app_ids):
    assert client.post("/recommend", json={"app_ids": app_ids}).status_code == 422


def test_recommend_rejects_unknown_games(client):
    response = client.post("/recommend", json={"app_ids": [1, 2, 999]})

    assert response.status_code == 422
    assert "999" in response.json()["detail"]


def test_home_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Find your next game" in response.text


def test_recommendations_page_and_shared_files_are_served(client):
    page = client.get("/recommendations")
    assert page.status_code == 200
    assert "Get recommendations" in page.text

    # Both pages load these two shared files.
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/common.js").status_code == 200


def test_top_returns_most_liked_real_games_with_three_tags(client):
    response = client.get("/top", params={"limit": 2})

    assert response.status_code == 200
    games = response.json()
    # Toy likes: Hollow Knight 12, Hades 11 (log scale).
    assert [game["title"] for game in games] == ["Hollow Knight", "Hades"]
    assert games[0]["tags"] == ["Metroidvania", "Souls-like", "Platformer"]
    assert games[0]["positive_ratio"] == 97
    assert games[0]["user_reviews"] == 300000


def test_top_validates_the_limit(client):
    assert client.get("/top", params={"limit": 0}).status_code == 422
    assert client.get("/top", params={"limit": 51}).status_code == 422
