"""Turn any retriever's neighbour table into ranked candidates per user (D21, D23).

A retriever (co-occurrence, content) provides a table app_id -> neighbor_id with a similarity.
For a user's seed games, a candidate's score is the sum of its similarity to each seed, and the
seed contributing most is kept as the explanation. Seeds themselves are never recommended.
"""

import pandas as pd

TOP_N = 200


def remove_seeds(candidates: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Drop candidate rows whose (user_id, app_id) is one of that user's seed games."""
    matched = candidates.merge(seeds[["user_id", "app_id"]], on=["user_id", "app_id"], how="left", indicator=True)
    is_seed = (matched["_merge"] == "both").to_numpy()
    return candidates[~is_seed]


def keep_top_n(candidates: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Sort each user's candidates by score (ties by app_id), keep the best `top_n`, add rank 1..top_n."""
    ranked = candidates.sort_values(["user_id", "score", "app_id"], ascending=[True, False, True])
    ranked = ranked.groupby("user_id").head(top_n).copy()
    ranked["rank"] = ranked.groupby("user_id").cumcount() + 1
    return ranked.reset_index(drop=True)


def recommend_from_neighbors(seeds: pd.DataFrame, neighbors: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Rank candidates per user by summed similarity to their seeds; returns user_id, app_id, score, best_seed, rank."""
    # One row per (user, seed, neighbour of that seed).
    seed_neighbors = seeds[["user_id", "app_id"]].merge(neighbors[["app_id", "neighbor_id", "similarity"]], on="app_id")
    seed_neighbors = seed_neighbors.rename(columns={"app_id": "seed_id", "neighbor_id": "app_id"})
    candidates = remove_seeds(seed_neighbors, seeds)

    # Score = sum of the candidate's similarity to each seed.
    scores = candidates.groupby(["user_id", "app_id"], as_index=False).agg(score=("similarity", "sum"))

    # Explanation = the seed with the highest similarity (ties: lowest seed_id, so it's always the same).
    best = candidates.sort_values(["similarity", "seed_id"], ascending=[False, True])
    best = best.drop_duplicates(subset=["user_id", "app_id"], keep="first")
    best = best[["user_id", "app_id", "seed_id"]].rename(columns={"seed_id": "best_seed"})

    recs = scores.merge(best, on=["user_id", "app_id"])
    return keep_top_n(recs, top_n)
