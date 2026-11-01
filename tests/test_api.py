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
