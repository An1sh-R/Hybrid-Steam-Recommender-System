"""Item-item co-occurrence: for every game, the games most often liked by the same players.

For each training cutoff (val, test) this stage:
1. keeps likes before the cutoff, at most each user's most recent MAX_LIKES_PER_USER (D10, D20)
2. counts, for every pair of games, how many users liked both (`both_liked`)
3. scores each pair with cosine similarity: both_liked / sqrt(likes_a * likes_b)
4. keeps each game's TOP_K most similar games

Outputs data/cooccurrence/{val,test}_neighbors.parquet with columns
app_id, neighbor_id, both_liked, similarity, also_liked_share.
`also_liked_share` = both_liked / likes_a: "N% of players who liked A also liked B" (D11).

Run with: uv run python -m steam_rec.cooccurrence
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from steam_rec.data import PROJECT_ROOT
from steam_rec.prepare import PROCESSED_DIR
from steam_rec.split import TEST_CUTOFF, VAL_CUTOFF

COOCCURRENCE_DIR = PROJECT_ROOT / "data" / "cooccurrence"

# Neighbours for the val set learn from likes before VAL_CUTOFF; for test, before TEST_CUTOFF.
FIT_CUTOFFS = {"val": VAL_CUTOFF, "test": TEST_CUTOFF}

MAX_LIKES_PER_USER = 100
MIN_BOTH_LIKED = 5
TOP_K = 100
CHUNK_SIZE = 500


def likes_before(reviews: pd.DataFrame, cutoff: pd.Timestamp, max_per_user: int = MAX_LIKES_PER_USER) -> pd.DataFrame:
    """Return (user_id, app_id) likes before `cutoff`, keeping each user's most recent `max_per_user`."""
    likes = reviews[reviews["is_recommended"] & (reviews["date"] < cutoff)]
    likes = likes.sort_values("date")
    likes = likes.groupby("user_id").tail(max_per_user)

    # A user with a single like can't form a pair, so they add nothing.
    likes_per_user = likes["user_id"].value_counts()
    users_with_pairs = likes_per_user[likes_per_user >= 2].index
    likes = likes[likes["user_id"].isin(users_with_pairs)]
    return likes[["user_id", "app_id"]].reset_index(drop=True)


def build_like_matrix(likes: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return a users x games sparse matrix (1 = liked) and the app_id of each column."""
    user_codes, user_ids = pd.factorize(likes["user_id"])
    game_codes, app_ids = pd.factorize(likes["app_id"])
    ones = np.ones(len(likes), dtype=np.float32)
    matrix = sparse.csr_matrix((ones, (user_codes, game_codes)), shape=(len(user_ids), len(app_ids)))
    return matrix, np.asarray(app_ids)


def top_neighbors(
    matrix: sparse.csr_matrix,
    app_ids: np.ndarray,
    top_k: int = TOP_K,
    min_both_liked: int = MIN_BOTH_LIKED,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """For every game, return its `top_k` most similar games by cosine similarity."""
    likes_per_game = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    games_by_users = matrix.T.tocsr()
    n_games = matrix.shape[1]

    # Work on `chunk_size` games at a time so the pair counts never all sit in memory at once.
    chunks = []
    for start in range(0, n_games, chunk_size):
        end = min(start + chunk_size, n_games)
        # both_liked[a, b] = number of users who liked game a and game b.
        both_liked = (games_by_users[start:end] @ matrix).tocoo()

        pairs = pd.DataFrame({
            "game": both_liked.row + start,
            "neighbor": both_liked.col,
            "both_liked": both_liked.data.astype(np.int32),
        })
        not_itself = pairs["game"] != pairs["neighbor"]
        enough_users = pairs["both_liked"] >= min_both_liked
        pairs = pairs[not_itself & enough_users]

        game_likes = likes_per_game[pairs["game"].to_numpy()]
        neighbor_likes = likes_per_game[pairs["neighbor"].to_numpy()]
        pairs["similarity"] = pairs["both_liked"] / np.sqrt(game_likes * neighbor_likes)
        pairs["also_liked_share"] = pairs["both_liked"] / game_likes

        pairs = pairs.sort_values(["game", "similarity"], ascending=[True, False])
        pairs = pairs.groupby("game").head(top_k)
        chunks.append(pairs)

    neighbors = pd.concat(chunks, ignore_index=True)
    neighbors["app_id"] = app_ids[neighbors["game"].to_numpy()]
    neighbors["neighbor_id"] = app_ids[neighbors["neighbor"].to_numpy()]
    columns = ["app_id", "neighbor_id", "both_liked", "similarity", "also_liked_share"]
    return neighbors[columns].reset_index(drop=True)


def build_neighbors(reviews: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Run the three steps (likes, matrix, neighbours) for one training cutoff."""
    likes = likes_before(reviews, cutoff)
    matrix, app_ids = build_like_matrix(likes)
    return top_neighbors(matrix, app_ids)


def cooccurrence(processed_dir: Path = PROCESSED_DIR, out_dir: Path = COOCCURRENCE_DIR) -> None:
    """Build and save the neighbour table for each training cutoff."""
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cutoff in FIT_CUTOFFS.items():
        neighbors = build_neighbors(reviews, cutoff)
        neighbors.to_parquet(out_dir / f"{name}_neighbors.parquet", index=False)
        n_games = neighbors["app_id"].nunique()
        print(f"{name}: {n_games:,} games with neighbours, {len(neighbors):,} rows")


if __name__ == "__main__":
    cooccurrence()
