"""Tests for steam_rec.prepare using the tiny raw files from the `raw_dir` fixture (conftest.py)."""

import pandas as pd
import pytest

from steam_rec.prepare import build_catalog, prepare


def test_build_catalog_adds_description_and_tags():
    games = pd.DataFrame({"app_id": [10, 20], "title": ["A", "B"]})
    meta = pd.DataFrame({
        "app_id": [20, 10],
        "description": ["about B", "about A"],
        "tags": [["Puzzle"], ["RPG", "Indie"]],
    })
    catalog = build_catalog(games, meta)

    assert catalog["title"].tolist() == ["A", "B"]
    assert catalog["description"].tolist() == ["about A", "about B"]
    assert catalog["tags"].tolist() == [["RPG", "Indie"], ["Puzzle"]]


def test_build_catalog_rejects_duplicate_metadata():
    games = pd.DataFrame({"app_id": [10], "title": ["A"]})
    meta = pd.DataFrame({"app_id": [10, 10], "description": ["x", "y"], "tags": [[], []]})

    with pytest.raises(pd.errors.MergeError):
        build_catalog(games, meta)


def test_prepare_writes_parquet_files_that_read_back(raw_dir, tmp_path):
    out_dir = tmp_path / "processed"
    prepare(raw_dir, out_dir)

    catalog = pd.read_parquet(out_dir / "catalog.parquet")
    reviews = pd.read_parquet(out_dir / "reviews.parquet")

    assert len(catalog) == 2
    assert list(catalog.loc[0, "tags"]) == ["Metroidvania", "Souls-like"]
    assert "is_non_game" in catalog.columns
    assert len(reviews) == 2
    assert reviews["user_id"].dtype == "int32"
