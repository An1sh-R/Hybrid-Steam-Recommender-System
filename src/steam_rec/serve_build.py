"""Build the files the app needs, using all the data (decision D28).

Everything before this was fitted on data before a cutoff, to evaluate honestly. The app should
use everything we know, so this stage refits on all reviews (FULL_CUTOFF is the day after the
last review) and writes small files to data/serve/:
- cooc_neighbors.parquet: co-occurrence neighbours from all likes
- game_features.parquet: the ranker's per-game features as of FULL_CUTOFF
- games.parquet: title, release year, tags, price, ... for search and display

The content neighbours (data/content) and the ranker model (models/ranker.txt, trained on the
val period) are used as they are. The app never needs the 2GB of reviews.
Run with: uv run python -m steam_rec.serve_build
"""

from pathlib import Path

import pandas as pd

from steam_rec.cooccurrence import build_neighbors
from steam_rec.data import PROJECT_ROOT
from steam_rec.prepare import PROCESSED_DIR
from steam_rec.ranker import game_features

SERVE_DIR = PROJECT_ROOT / "data" / "serve"

# The last review is on 2022-12-31, so "before 2023-01-01" means all of them.
FULL_CUTOFF = pd.Timestamp("2023-01-01")

GAME_COLUMNS = ["app_id", "title", "release_year", "tags", "is_non_game", "price_final", "positive_ratio", "user_reviews"]


def serving_games(catalog: pd.DataFrame) -> pd.DataFrame:
    """The catalog columns the app shows or searches, with the release date reduced to a year."""
    games = catalog.copy()
    games["release_year"] = games["date_release"].dt.year
    return games[GAME_COLUMNS].reset_index(drop=True)


def serve_build(processed_dir: Path = PROCESSED_DIR, out_dir: Path = SERVE_DIR) -> None:
    """Refit co-occurrence and game features on all data and write the app's files."""
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    catalog = pd.read_parquet(processed_dir / "catalog.parquet")

    cooc_neighbors = build_neighbors(reviews, FULL_CUTOFF)
    features = game_features(reviews, catalog, FULL_CUTOFF)
    del reviews
    games = serving_games(catalog)

    out_dir.mkdir(parents=True, exist_ok=True)
    cooc_neighbors.to_parquet(out_dir / "cooc_neighbors.parquet", index=False)
    features.to_parquet(out_dir / "game_features.parquet", index=False)
    games.to_parquet(out_dir / "games.parquet", index=False)

    print(f"cooc_neighbors: {cooc_neighbors['app_id'].nunique():,} games, {len(cooc_neighbors):,} rows")
    print(f"game_features: {len(features):,} games; games: {len(games):,} rows")


if __name__ == "__main__":
    serve_build()
