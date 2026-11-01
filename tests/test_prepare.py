"""Tests for steam_rec.prepare using the tiny raw files from the `raw_dir` fixture (conftest.py)."""

import json

import pandas as pd
import pytest

from steam_rec.prepare import build_catalog, fill_missing_tags, prepare


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


def test_fill_missing_tags_only_fills_empty_lists():
    meta = pd.DataFrame({
        "app_id": [10, 20, 30],
        "tags": [["RPG"], [], []],
    })
    steamspy_tags = pd.DataFrame({
        "app_id": [10, 20],
        "tags": [["Should", "Not", "Replace"], ["FPS", "Shooter"]],
    })
    filled = fill_missing_tags(meta, steamspy_tags)

    # 10 keeps its Kaggle tags, 20 is filled from SteamSpy, 30 has no SteamSpy entry so stays empty.
    assert filled["tags"].tolist() == [["RPG"], ["FPS", "Shooter"], []]
    # The input table is left unchanged.
    assert meta["tags"].tolist() == [["RPG"], [], []]


def test_prepare_writes_parquet_files_that_read_back(raw_dir, tmp_path):
    steamspy_path = tmp_path / "steamspy_tags.jsonl"
    steamspy_path.write_text(json.dumps({"app_id": 20, "tags": ["Soundtrack"]}) + "\n", encoding="utf-8")
    out_dir = tmp_path / "processed"
    prepare(raw_dir, out_dir, steamspy_path)

    catalog = pd.read_parquet(out_dir / "catalog.parquet")
    reviews = pd.read_parquet(out_dir / "reviews.parquet")

    assert len(catalog) == 2
    assert list(catalog.loc[0, "tags"]) == ["Metroidvania", "Souls-like"]
    assert list(catalog.loc[1, "tags"]) == ["Soundtrack"]
    assert "is_non_game" in catalog.columns
    assert len(reviews) == 2
    assert reviews["user_id"].dtype == "int32"
