"""Content-based neighbours: for every game, the games with the most similar Steam tags.

Each game becomes a vector over the 441 tags using TF-IDF: a tag the game has gets weight
log(N / games_with_tag) (smoothed), so a rare tag like "Metroidvania" counts far more than
"Indie", which half the catalog has. Each vector is scaled to length 1, so the dot product of
two games' vectors is their cosine similarity. Each game keeps its TOP_K most similar games.

Unlike co-occurrence there is one table, not one per cutoff: tags come from the catalog, not
from reviews, and every catalog game has them (even games nobody reviewed yet).
Non-games (soundtracks, DLC, ...) are never neighbours (D10).

Outputs data/content/neighbors.parquet with columns app_id, neighbor_id, similarity.
Run with: uv run python -m steam_rec.content
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from steam_rec.data import PROJECT_ROOT
from steam_rec.prepare import PROCESSED_DIR

CONTENT_DIR = PROJECT_ROOT / "data" / "content"

TOP_K = 100
CHUNK_SIZE = 500


def build_tag_matrix(tags: pd.Series) -> np.ndarray:
    """Return a games x tags TF-IDF array, one row per game, each row of length 1 (or all zeros if no tags)."""
    tag_lists = []
    for game_tags in tags:
        tag_lists.append(list(game_tags))

    # has_tag[g, t] = 1 if game g has tag t.
    binarizer = MultiLabelBinarizer(sparse_output=True)
    has_tag = binarizer.fit_transform(tag_lists)

    tfidf = TfidfTransformer(norm="l2", smooth_idf=True).fit_transform(has_tag)
    return tfidf.toarray().astype(np.float32)


def top_content_neighbors(
    tfidf: np.ndarray,
    app_ids: np.ndarray,
    can_be_neighbor: np.ndarray,
    top_k: int = TOP_K,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """For every game, its `top_k` most tag-similar games among those where `can_be_neighbor` is True."""
    n_games = tfidf.shape[0]
    chunks = []
    for start in range(0, n_games, chunk_size):
        end = min(start + chunk_size, n_games)

        # Rows have length 1, so a dot product is the cosine similarity. Shape: chunk x all games.
        similarity = tfidf[start:end] @ tfidf.T
        similarity[:, ~can_be_neighbor] = 0.0
        for row in range(end - start):
            similarity[row, start + row] = 0.0  # a game is not its own neighbour

        # For each row, the columns of its top_k similarities (stable sort: ties keep catalog order).
        best_columns = np.argsort(-similarity, axis=1, kind="stable")[:, :top_k]
        best_values = np.take_along_axis(similarity, best_columns, axis=1)

        n_kept = best_columns.shape[1]
        chunk = pd.DataFrame({
            "app_id": np.repeat(app_ids[start:end], n_kept),
            "neighbor_id": app_ids[best_columns.ravel()],
            "similarity": best_values.ravel(),
        })
        # Games sharing no tag at all have similarity 0: not neighbours.
        chunks.append(chunk[chunk["similarity"] > 0])

    return pd.concat(chunks, ignore_index=True)


def content(processed_dir: Path = PROCESSED_DIR, out_dir: Path = CONTENT_DIR) -> None:
    """Build and save the tag-similarity neighbour table for the whole catalog."""
    catalog = pd.read_parquet(processed_dir / "catalog.parquet", columns=["app_id", "tags", "is_non_game"])
    tfidf = build_tag_matrix(catalog["tags"])
    app_ids = catalog["app_id"].to_numpy()
    can_be_neighbor = ~catalog["is_non_game"].to_numpy()

    neighbors = top_content_neighbors(tfidf, app_ids, can_be_neighbor)

    out_dir.mkdir(parents=True, exist_ok=True)
    neighbors.to_parquet(out_dir / "neighbors.parquet", index=False)
    print(f"content: {neighbors['app_id'].nunique():,} games with neighbours, {len(neighbors):,} rows")


if __name__ == "__main__":
    content()
