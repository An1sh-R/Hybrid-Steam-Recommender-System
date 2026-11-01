"""Turn a few games a person liked into 10 recommendations with explanations (decisions D11, D27-D29).

The same steps that were evaluated, for one person:
1. both retrievers propose candidates from the seed games (`build_candidates`)
2. the ranker scores them using the per-game features as of the end of the data
3. keep the ranker's top 200, apply the popularity penalty (alpha = 0.3), keep the top 10
4. explain each one with the retriever that ranked it higher:
   - co-occurrence: "Players who liked X also liked this"
   - content:       "Because you liked X: shares tag1, tag2, tag3"

Also `search_games` for finding a game by title, and `load_artifacts` to read data/serve etc.
"""

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from steam_rec.candidates import TOP_N, keep_top_n
from steam_rec.content import CONTENT_DIR
from steam_rec.ranker import FEATURES, MODEL_PATH, build_candidates
from steam_rec.rerank import popularity_penalty
from steam_rec.serve_build import SERVE_DIR

ALPHA = 0.3
N_RECOMMENDATIONS = 10
N_SHARED_TAGS = 3


@dataclass
class Artifacts:
    """Everything recommend() needs, loaded once when the app starts."""

    games: pd.DataFrame
    cooc_neighbors: pd.DataFrame
    content_neighbors: pd.DataFrame
    game_features: pd.DataFrame
    model: lgb.Booster
    max_log_likes: float


def load_artifacts(
    serve_dir: Path = SERVE_DIR, content_dir: Path = CONTENT_DIR, model_path: Path = MODEL_PATH
) -> Artifacts:
    """Read the serving files and the ranker model from disk."""
    game_features = pd.read_parquet(serve_dir / "game_features.parquet")
    return Artifacts(
        games=pd.read_parquet(serve_dir / "games.parquet"),
        cooc_neighbors=pd.read_parquet(serve_dir / "cooc_neighbors.parquet"),
        content_neighbors=pd.read_parquet(content_dir / "neighbors.parquet"),
        game_features=game_features,
        model=lgb.Booster(model_file=str(model_path)),
        max_log_likes=float(game_features["log_likes"].max()),
    )


def search_games(query: str, games: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Real games whose title contains `query` (any case), most-reviewed first."""
    matches = games["title"].str.contains(query, case=False, regex=False)
    found = games[matches & ~games["is_non_game"]]
    found = found.sort_values("user_reviews", ascending=False)
    return found.head(limit)[["app_id", "title", "release_year"]]


def explain(row: pd.Series, games_by_id: pd.DataFrame) -> dict:
    """The reason for one recommendation: which seed led to it, and why."""
    # The retriever that ranked the game higher explains it; on a tie, co-occurrence.
    if row["cooc_rank"] <= row["content_rank"]:
        seed_id = int(row["cooc_best_seed"])
        seed_title = games_by_id.loc[seed_id, "title"]
        return {"type": "players_also_liked", "seed_app_id": seed_id, "shared_tags": [],
                "text": f"Players who liked {seed_title} also liked this"}

    seed_id = int(row["content_best_seed"])
    seed_title = games_by_id.loc[seed_id, "title"]
    seed_tags = set(games_by_id.loc[seed_id, "tags"])
    # Keep the recommended game's own tag order (most-voted first).
    shared_tags = []
    for tag in games_by_id.loc[row["app_id"], "tags"]:
        if tag in seed_tags:
            shared_tags.append(tag)
    shared_tags = shared_tags[:N_SHARED_TAGS]

    if shared_tags:
        text = f"Because you liked {seed_title}: shares {', '.join(shared_tags)}"
    else:
        text = f"Similar to {seed_title}"
    return {"type": "similar_tags", "seed_app_id": seed_id, "shared_tags": shared_tags, "text": text}


def score_and_rerank(
    candidates: pd.DataFrame, model: lgb.Booster, max_log_likes: float, alpha: float = ALPHA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranker scores -> each user's top 200 (`ranked`) -> popularity penalty -> each user's top 10 (`final`)."""
    scored = candidates.copy()
    scored["score"] = model.predict(candidates[FEATURES])
    ranked = keep_top_n(scored, TOP_N)
    final = popularity_penalty(ranked, alpha, max_log_likes)
    return ranked, final


def recommend(
    seed_ids: list[int], artifacts: Artifacts, n: int = N_RECOMMENDATIONS, alpha: float = ALPHA
) -> list[dict]:
    """Recommend `n` games for someone who liked the games in `seed_ids`, each with an explanation."""
    games_by_id = artifacts.games.set_index("app_id")
    unknown = []
    for app_id in seed_ids:
        if app_id not in games_by_id.index:
            unknown.append(app_id)
    if unknown:
        raise ValueError(f"unknown app_ids: {unknown}")

    # One "user" (id 0) with these seeds. Only the seeds' neighbour rows are needed.
    seeds = pd.DataFrame({"user_id": 0, "app_id": seed_ids})
    cooc = artifacts.cooc_neighbors[artifacts.cooc_neighbors["app_id"].isin(seed_ids)]
    content = artifacts.content_neighbors[artifacts.content_neighbors["app_id"].isin(seed_ids)]
    candidates = build_candidates(seeds, cooc, content)
    if len(candidates) == 0:
        return []

    # Never recommend soundtracks / DLC (D10), whichever retriever found them.
    non_game_ids = artifacts.games.loc[artifacts.games["is_non_game"], "app_id"]
    candidates = candidates[~candidates["app_id"].isin(non_game_ids)]

    candidates = candidates.merge(artifacts.game_features, on="app_id", how="left")
    ranked, final = score_and_rerank(candidates, artifacts.model, artifacts.max_log_likes, alpha)
    final = final.head(n)
    # Bring back the candidate columns (ranks, best seeds) that the explanation needs.
    final = final[["app_id", "rank"]].merge(ranked.drop(columns=["score", "rank"]), on="app_id")

    recommendations = []
    for _, row in final.sort_values("rank").iterrows():
        game = games_by_id.loc[row["app_id"]]
        recommendations.append({
            "app_id": int(row["app_id"]),
            "title": game["title"],
            "release_year": int(game["release_year"]),
            "price": float(game["price_final"]),
            "reason": explain(row, games_by_id),
        })
    return recommendations
