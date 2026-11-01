"""Fetch Steam community tags from SteamSpy for catalog games whose tags are missing (D24).

The Kaggle metadata has empty tags for 1,244 games, mostly the most popular ones (D23).
SteamSpy's appdetails endpoint returns the same community tags with their vote counts.
Non-games are skipped (never recommended anyway), and the rest are fetched most-popular first:
SteamSpy is slow and often drops connections, so if a run is cut short, the games that matter
most are already done. Each result is appended to a JSON-lines file as it arrives, so an
interrupted run resumes where it stopped; a game that keeps failing is skipped and retried next run.

Output: data/external/steamspy_tags.jsonl, one line per game: {"app_id": ..., "tags": [...]}.
It is fetched once and versioned with `dvc add data/external`, not re-fetched by the pipeline.
Run with: uv run python -m steam_rec.fetch_tags
"""

import json
import time
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

from steam_rec.data import RAW_DIR, STEAMSPY_TAGS_PATH, load_games, load_metadata

STEAMSPY_URL = "https://steamspy.com/api.php?request=appdetails&appid={app_id}"
USER_AGENT = "steam-rec student project (github.com/An1sh-R/Hybrid-Steam-Recommender-System)"
PAUSE_SECONDS = 1.0
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
RETRY_WAIT_SECONDS = 10


def parse_tags(response: dict) -> list[str]:
    """Return the tag names from a SteamSpy appdetails response, most-voted first."""
    tags = response.get("tags", {})
    # SteamSpy sends an empty list instead of a dict when a game has no tags.
    if not isinstance(tags, dict):
        return []
    return sorted(tags, key=tags.get, reverse=True)


def fetch_steamspy(app_id: int) -> dict | None:
    """Download one game's SteamSpy appdetails, retrying on network errors; None if every attempt fails."""
    url = STEAMSPY_URL.format(app_id=app_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return json.load(response)
        except (OSError, ValueError) as error:
            print(f"app {app_id}: attempt {attempt} failed ({error})")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS * attempt)
    return None


def missing_tag_ids(meta: pd.DataFrame, games: pd.DataFrame) -> list[int]:
    """Return the app_ids of real games (not soundtracks/DLC) with no tags, most-reviewed on Steam first."""
    no_tags = meta.loc[meta["tags"].str.len() == 0, ["app_id"]]
    no_tags = no_tags.merge(games[["app_id", "is_non_game", "user_reviews"]], on="app_id")
    no_tags = no_tags[~no_tags["is_non_game"]]
    no_tags = no_tags.sort_values("user_reviews", ascending=False)
    return no_tags["app_id"].tolist()


def already_fetched(path: Path) -> set[int]:
    """Return the app_ids already saved in the output file (empty if it doesn't exist yet)."""
    if not path.exists():
        return set()
    fetched = set()
    with open(path, encoding="utf-8") as saved:
        for line in saved:
            fetched.add(json.loads(line)["app_id"])
    return fetched


def ids_to_fetch(raw_dir: Path, out_path: Path) -> list[int]:
    """Return the tagless real games not yet saved in `out_path`, most popular first."""
    # The tables only live inside this function, so their memory is freed once the list is built.
    meta = load_metadata(raw_dir)
    games = load_games(raw_dir)
    done = already_fetched(out_path)
    to_fetch = []
    for app_id in missing_tag_ids(meta, games):
        if app_id not in done:
            to_fetch.append(app_id)
    print(f"{len(done):,} already fetched, {len(to_fetch):,} to go")
    return to_fetch


def fetch_tags(
    raw_dir: Path = RAW_DIR,
    out_path: Path = STEAMSPY_TAGS_PATH,
    fetch_one: Callable[[int], dict | None] = fetch_steamspy,
    pause: float = PAUSE_SECONDS,
) -> None:
    """Fetch SteamSpy tags for every tagless game not already in `out_path`, appending as we go."""
    to_fetch = ids_to_fetch(raw_dir, out_path)

    skipped = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out_file:
        for count, app_id in enumerate(to_fetch, start=1):
            response = fetch_one(app_id)
            if response is None:
                # Not saved, so the next run will try this game again.
                skipped.append(app_id)
            else:
                record = {"app_id": int(app_id), "tags": parse_tags(response)}
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
            if count % 100 == 0:
                print(f"{count:,} / {len(to_fetch):,}")
            time.sleep(pause)

    print(f"done; {len(skipped):,} games failed every attempt and will be retried next run: {skipped}")


if __name__ == "__main__":
    fetch_tags()
