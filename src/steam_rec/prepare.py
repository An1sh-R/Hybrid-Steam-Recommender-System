"""Turn the raw Kaggle files into cleaned parquet files in data/processed/.

Outputs:
- catalog.parquet: one row per game, games.csv joined with its description and tags.
  Games whose Kaggle tags are empty get their tags from SteamSpy instead (D24).
- reviews.parquet: cleaned reviews (see steam_rec.data)

Run with: uv run python -m steam_rec.prepare
This is the first stage of the DVC pipeline (decision D14).
"""

from pathlib import Path

import pandas as pd

from steam_rec.data import (
    PROJECT_ROOT,
    RAW_DIR,
    STEAMSPY_TAGS_PATH,
    load_games,
    load_metadata,
    load_reviews,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_steamspy_tags(path: Path = STEAMSPY_TAGS_PATH) -> pd.DataFrame:
    """Load the fetched SteamSpy tags: one row per game with columns app_id, tags."""
    return pd.read_json(path, lines=True)


def fill_missing_tags(meta: pd.DataFrame, steamspy_tags: pd.DataFrame) -> pd.DataFrame:
    """Replace empty tag lists in `meta` with SteamSpy's tags for the same game, where available."""
    steamspy_by_app = dict(zip(steamspy_tags["app_id"], steamspy_tags["tags"], strict=True))

    filled_tags = []
    for app_id, tags in zip(meta["app_id"], meta["tags"], strict=True):
        if len(tags) == 0 and app_id in steamspy_by_app:
            filled_tags.append(list(steamspy_by_app[app_id]))
        else:
            filled_tags.append(tags)

    meta = meta.copy()
    meta["tags"] = filled_tags
    return meta


def build_catalog(games: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Join each game with its description and tags (exactly one metadata row per game)."""
    catalog = games.merge(meta, on="app_id", how="left", validate="one_to_one")
    return catalog


def prepare(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    steamspy_path: Path = STEAMSPY_TAGS_PATH,
) -> None:
    """Load and clean the raw files, fill missing tags, then save catalog.parquet and reviews.parquet."""
    games = load_games(raw_dir)
    meta = load_metadata(raw_dir)
    meta = fill_missing_tags(meta, load_steamspy_tags(steamspy_path))
    catalog = build_catalog(games, meta)
    reviews = load_reviews(raw_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)
    reviews.to_parquet(out_dir / "reviews.parquet", index=False)

    games_without_tags = (catalog["tags"].str.len() == 0).sum()
    print(f"catalog: {len(catalog):,} games ({games_without_tags:,} still without tags)")
    print(f"reviews: {len(reviews):,} rows -> {out_dir / 'reviews.parquet'}")


if __name__ == "__main__":
    prepare()
