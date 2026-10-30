"""Load the raw Kaggle files into clean DataFrames.

Applies the cleaning rules from the EDA (decisions D10):
- games: drop the constant `steam_deck` column, flag soundtrack/DLC-like titles
- reviews: small dtypes to fit 41M rows in memory, keep only the columns we use,
  drop duplicate (user, game) reviews
`users.csv` is not loaded: it adds nothing beyond the reviews file.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Whole words in a title that mark it as not a game.
NON_GAME_WORDS = ["Soundtrack", "OST", "DLC", "Season Pass", "Artbook"]

REVIEW_COLUMNS = ["user_id", "app_id", "date", "is_recommended", "hours"]
REVIEW_DTYPES = {
    "user_id": "int32",
    "app_id": "int32",
    "is_recommended": "bool",
    "hours": "float32",
}


def flag_non_games(titles: pd.Series) -> pd.Series:
    """Return True for titles that look like soundtracks, DLC, season passes or artbooks."""
    looks_non_game = pd.Series(False, index=titles.index)
    for word in NON_GAME_WORDS:
        # \b = word boundary, so "OST" does not match "Ghost" or "Lost".
        pattern = rf"\b{word}\b"
        looks_non_game = looks_non_game | titles.str.contains(pattern, regex=True)
    return looks_non_game


def load_games(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load games.csv, drop `steam_deck` and add an `is_non_game` flag."""
    games = pd.read_csv(raw_dir / "games.csv", parse_dates=["date_release"])
    games = games.drop(columns=["steam_deck"])
    games["is_non_game"] = flag_non_games(games["title"])
    return games


def load_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load games_metadata.json: app_id, description and a list of tags per game."""
    return pd.read_json(raw_dir / "games_metadata.json", lines=True)


def load_reviews(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load recommendations.csv with small dtypes, keeping the latest review per (user, game)."""
    reviews = pd.read_csv(
        raw_dir / "recommendations.csv",
        usecols=REVIEW_COLUMNS,
        dtype=REVIEW_DTYPES,
        parse_dates=["date"],
    )
    reviews = reviews.sort_values("date")
    reviews = reviews.drop_duplicates(subset=["user_id", "app_id"], keep="last")
    reviews = reviews.reset_index(drop=True)
    return reviews
