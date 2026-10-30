"""Turn the raw Kaggle files into cleaned parquet files in data/processed/.

Outputs:
- catalog.parquet: one row per game, games.csv joined with its description and tags
- reviews.parquet: cleaned reviews (see steam_rec.data)

Run with: uv run python -m steam_rec.prepare
This is the first stage of the DVC pipeline (decision D14).
"""

from pathlib import Path

import pandas as pd

from steam_rec.data import PROJECT_ROOT, RAW_DIR, load_games, load_metadata, load_reviews

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_catalog(games: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Join each game with its description and tags (exactly one metadata row per game)."""
    catalog = games.merge(meta, on="app_id", how="left", validate="one_to_one")
    return catalog


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> None:
    """Load and clean the raw files, then save catalog.parquet and reviews.parquet."""
    games = load_games(raw_dir)
    meta = load_metadata(raw_dir)
    catalog = build_catalog(games, meta)
    reviews = load_reviews(raw_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)
    reviews.to_parquet(out_dir / "reviews.parquet", index=False)

    print(f"catalog: {len(catalog):,} games -> {out_dir / 'catalog.parquet'}")
    print(f"reviews: {len(reviews):,} rows -> {out_dir / 'reviews.parquet'}")


if __name__ == "__main__":
    prepare()
