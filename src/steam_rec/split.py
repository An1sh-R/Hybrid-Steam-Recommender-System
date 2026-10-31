"""Build the evaluation sets: seed games and target games per user, split by time.

Time periods (decisions D4, D10, D19):
- train: date < VAL_CUTOFF                  -> what retrievers learn from for the val set
- val:   VAL_CUTOFF <= date < TEST_CUTOFF   -> targets for training and tuning the ranker
- test:  date >= TEST_CUTOFF                -> targets for the final evaluation

For val and test, an evaluation user needs at least MIN_SEEDS liked games before the period
(their most recent MAX_SEEDS become the seeds, like the 3-5 games typed into the app)
and at least one liked game inside the period (the targets).

Outputs in data/split/: val_seeds, val_targets, test_seeds, test_targets (.parquet).
Run with: uv run python -m steam_rec.split
"""

from pathlib import Path

import numpy as np
import pandas as pd

from steam_rec.data import PROJECT_ROOT
from steam_rec.prepare import PROCESSED_DIR

SPLIT_DIR = PROJECT_ROOT / "data" / "split"

# The 80% and 90% quantiles of review dates (notebooks/01_eda.ipynb).
VAL_CUTOFF = pd.Timestamp("2022-02-16")
TEST_CUTOFF = pd.Timestamp("2022-08-13")

MIN_SEEDS = 3
MAX_SEEDS = 5
MAX_USERS = 50_000
RANDOM_SEED = 42


def sample_users(user_ids: np.ndarray, max_users: int, random_seed: int) -> np.ndarray:
    """Return at most `max_users` of `user_ids`, picked at random but reproducibly, sorted."""
    if len(user_ids) <= max_users:
        return np.sort(user_ids)
    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(user_ids, size=max_users, replace=False)
    return np.sort(sampled)


def build_eval_set(
    reviews: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    max_users: int = MAX_USERS,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (seeds, targets) for users who liked games both before `start` and in [start, end)."""
    positives = reviews[reviews["is_recommended"]]

    history = positives[positives["date"] < start]
    in_period = positives["date"] >= start
    if end is not None:
        in_period = in_period & (positives["date"] < end)
    period = positives[in_period]

    # Users with enough liked games before the period, and at least one liked game in it.
    seeds_per_user = history["user_id"].value_counts()
    users_with_seeds = seeds_per_user[seeds_per_user >= MIN_SEEDS].index
    users_with_targets = period["user_id"].unique()
    eval_users = users_with_seeds.intersection(users_with_targets).to_numpy()
    eval_users = sample_users(eval_users, max_users, random_seed)

    # Seeds: each user's most recent MAX_SEEDS liked games before the period.
    user_history = history[history["user_id"].isin(eval_users)]
    user_history = user_history.sort_values("date")
    seeds = user_history.groupby("user_id").tail(MAX_SEEDS)
    seeds = seeds[["user_id", "app_id"]]
    seeds = seeds.sort_values(["user_id", "app_id"]).reset_index(drop=True)

    # Targets: every game the user liked inside the period.
    # A target is a "new game" if nobody had liked it before the period (CF can't know it).
    targets = period[period["user_id"].isin(eval_users)]
    targets = targets[["user_id", "app_id"]].copy()
    games_liked_before = history["app_id"].unique()
    targets["is_new_game"] = ~targets["app_id"].isin(games_liked_before)
    targets = targets.sort_values(["user_id", "app_id"]).reset_index(drop=True)

    return seeds, targets


def split(processed_dir: Path = PROCESSED_DIR, out_dir: Path = SPLIT_DIR) -> None:
    """Build the val and test seed/target sets from reviews.parquet and save them."""
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    val_seeds, val_targets = build_eval_set(reviews, VAL_CUTOFF, TEST_CUTOFF)
    test_seeds, test_targets = build_eval_set(reviews, TEST_CUTOFF)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "val_seeds": val_seeds,
        "val_targets": val_targets,
        "test_seeds": test_seeds,
        "test_targets": test_targets,
    }
    for name, table in outputs.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)

    for name, targets in [("val", val_targets), ("test", test_targets)]:
        n_users = targets["user_id"].nunique()
        new_share = targets["is_new_game"].mean()
        print(f"{name}: {n_users:,} users, {len(targets):,} targets, {new_share:.1%} on new games")


if __name__ == "__main__":
    split()
