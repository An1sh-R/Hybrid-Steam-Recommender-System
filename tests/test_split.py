"""Tests for steam_rec.split on a small hand-built table of reviews."""

import pandas as pd

from steam_rec.split import build_eval_set, sample_users, split

START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2022-02-01")


def make_reviews(rows):
    """Build a reviews table from (user_id, app_id, date, is_recommended) tuples."""
    reviews = pd.DataFrame(rows, columns=["user_id", "app_id", "date", "is_recommended"])
    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews["hours"] = 10.0
    return reviews


def test_user_needs_enough_seeds_and_a_target():
    reviews = make_reviews([
        # user 1: 3 likes before, 1 like in the period -> evaluation user
        (1, 101, "2021-01-01", True),
        (1, 102, "2021-02-01", True),
        (1, 103, "2021-03-01", True),
        (1, 104, "2022-01-10", True),
        # user 2: only 2 likes before -> too few seeds
        (2, 101, "2021-01-01", True),
        (2, 102, "2021-02-01", True),
        (2, 104, "2022-01-10", True),
        # user 3: 3 likes before, but only a dislike in the period -> no target
        (3, 101, "2021-01-01", True),
        (3, 102, "2021-02-01", True),
        (3, 103, "2021-03-01", True),
        (3, 104, "2022-01-10", False),
        # user 4: dislikes before don't count as seeds
        (4, 101, "2021-01-01", True),
        (4, 102, "2021-02-01", True),
        (4, 103, "2021-03-01", False),
        (4, 104, "2022-01-10", True),
    ])
    seeds, targets = build_eval_set(reviews, START, END)

    assert seeds["user_id"].unique().tolist() == [1]
    assert seeds["app_id"].tolist() == [101, 102, 103]
    assert targets["user_id"].unique().tolist() == [1]
    assert targets["app_id"].tolist() == [104]


def test_seeds_are_the_most_recent_five_likes():
    rows = []
    for month in range(1, 8):
        rows.append((1, 100 + month, f"2021-{month:02d}-01", True))
    rows.append((1, 200, "2022-01-10", True))
    seeds, _ = build_eval_set(make_reviews(rows), START, END)

    assert seeds["app_id"].tolist() == [103, 104, 105, 106, 107]


def test_targets_stay_inside_the_period_and_flag_new_games():
    reviews = make_reviews([
        (1, 101, "2021-01-01", True),
        (1, 102, "2021-02-01", True),
        (1, 103, "2021-03-01", True),
        (2, 104, "2021-04-01", True),
        (1, 104, "2022-01-10", True),   # liked before by user 2 -> not new
        (1, 105, "2022-01-20", True),   # nobody liked it before -> new
        (1, 106, "2022-03-01", True),   # after END -> not a target
    ])
    _, targets = build_eval_set(reviews, START, END)

    assert targets["app_id"].tolist() == [104, 105]
    assert targets["is_new_game"].tolist() == [False, True]


def test_no_end_means_everything_after_start():
    reviews = make_reviews([
        (1, 101, "2021-01-01", True),
        (1, 102, "2021-02-01", True),
        (1, 103, "2021-03-01", True),
        (1, 106, "2023-06-01", True),
    ])
    _, targets = build_eval_set(reviews, START)

    assert targets["app_id"].tolist() == [106]


def test_sample_users_is_capped_and_reproducible():
    user_ids = pd.Series(range(100)).to_numpy()
    first = sample_users(user_ids, max_users=10, random_seed=42)
    second = sample_users(user_ids, max_users=10, random_seed=42)

    assert len(first) == 10
    assert first.tolist() == second.tolist()
    assert sample_users(user_ids, max_users=500, random_seed=42).tolist() == list(range(100))


def test_split_writes_the_four_files(tmp_path):
    reviews = make_reviews([
        (1, 101, "2021-01-01", True),
        (1, 102, "2021-02-01", True),
        (1, 103, "2021-03-01", True),
        (1, 104, "2022-03-01", True),
        (1, 105, "2022-09-01", True),
    ])
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    reviews.to_parquet(processed_dir / "reviews.parquet", index=False)

    out_dir = tmp_path / "split"
    split(processed_dir, out_dir)

    val_targets = pd.read_parquet(out_dir / "val_targets.parquet")
    test_targets = pd.read_parquet(out_dir / "test_targets.parquet")
    assert val_targets["app_id"].tolist() == [104]
    assert test_targets["app_id"].tolist() == [105]
    assert (out_dir / "val_seeds.parquet").exists()
    assert (out_dir / "test_seeds.parquet").exists()
