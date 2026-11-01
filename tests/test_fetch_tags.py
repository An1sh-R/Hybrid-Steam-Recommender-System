"""Tests for steam_rec.fetch_tags, using a fake SteamSpy instead of the network."""

import json

import pandas as pd

from steam_rec.fetch_tags import already_fetched, fetch_tags, missing_tag_ids, parse_tags

# Four games: 1 has tags; 2 and 3 are tagless real games (3 is more popular); 4 is a tagless soundtrack.
GAMES_CSV = """app_id,title,date_release,win,mac,linux,rating,positive_ratio,user_reviews,price_final,price_original,discount,steam_deck
1,Tagged Game,2020-01-01,true,false,false,Positive,90,500,9.99,9.99,0.0,true
2,Small Game,2020-01-01,true,false,false,Positive,90,50,9.99,9.99,0.0,true
3,Big Game,2020-01-01,true,false,false,Positive,90,90000,9.99,9.99,0.0,true
4,Big Game - Soundtrack,2020-01-01,true,false,false,Positive,90,40,4.99,4.99,0.0,true
"""

METADATA_JSON = """{"app_id":1,"description":"","tags":["RPG"]}
{"app_id":2,"description":"","tags":[]}
{"app_id":3,"description":"","tags":[]}
{"app_id":4,"description":"","tags":[]}
"""


def write_raw_files(folder):
    """Write the tiny games and metadata files used by fetch_tags."""
    (folder / "games.csv").write_text(GAMES_CSV)
    (folder / "games_metadata.json").write_text(METADATA_JSON)
    return folder


def fake_steamspy(app_id):
    """Pretend SteamSpy: every game has two tags, listed least-voted first."""
    return {"appid": app_id, "tags": {"Indie": 10, "Action": 50}}


def saved_records(path):
    """Read the JSON-lines output file back as a list of dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_tags_orders_by_votes():
    response = {"tags": {"Indie": 3863, "Metroidvania": 5623, "Souls-like": 4307}}
    assert parse_tags(response) == ["Metroidvania", "Souls-like", "Indie"]


def test_parse_tags_handles_no_tags():
    assert parse_tags({"tags": []}) == []
    assert parse_tags({}) == []


def test_missing_tag_ids_skips_non_games_and_puts_popular_first():
    meta = pd.DataFrame({"app_id": [1, 2, 3, 4], "tags": [["RPG"], [], [], []]})
    games = pd.DataFrame({
        "app_id": [1, 2, 3, 4],
        "is_non_game": [False, False, False, True],
        "user_reviews": [500, 50, 90000, 40],
    })
    assert missing_tag_ids(meta, games) == [3, 2]


def test_fetch_tags_writes_one_line_per_missing_game(tmp_path):
    raw_dir = write_raw_files(tmp_path)
    out_path = tmp_path / "external" / "steamspy_tags.jsonl"
    fetch_tags(raw_dir, out_path, fetch_one=fake_steamspy, pause=0)

    assert saved_records(out_path) == [
        {"app_id": 3, "tags": ["Action", "Indie"]},
        {"app_id": 2, "tags": ["Action", "Indie"]},
    ]


def test_fetch_tags_skips_games_already_saved(tmp_path):
    raw_dir = write_raw_files(tmp_path)
    out_path = tmp_path / "steamspy_tags.jsonl"
    out_path.write_text(json.dumps({"app_id": 3, "tags": ["Old"]}) + "\n", encoding="utf-8")

    def only_game_2(app_id):
        assert app_id == 2, f"game {app_id} should not be fetched again"
        return fake_steamspy(app_id)

    fetch_tags(raw_dir, out_path, fetch_one=only_game_2, pause=0)
    assert already_fetched(out_path) == {2, 3}


def test_a_game_that_keeps_failing_is_skipped_not_saved(tmp_path):
    raw_dir = write_raw_files(tmp_path)
    out_path = tmp_path / "steamspy_tags.jsonl"

    def game_3_always_fails(app_id):
        if app_id == 3:
            return None
        return fake_steamspy(app_id)

    fetch_tags(raw_dir, out_path, fetch_one=game_3_always_fails, pause=0)

    # Game 2 is still fetched; game 3 isn't saved, so a later run will retry it.
    assert already_fetched(out_path) == {2}
