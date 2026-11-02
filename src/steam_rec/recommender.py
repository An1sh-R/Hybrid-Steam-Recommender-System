"""Turn a few games a person liked into recommendations with explanations (decisions D11, D27-D29, D32).

The same steps that were evaluated, for one person:
1. both retrievers propose candidates from the seed games (`build_candidates`)
2. the ranker scores them using the per-game features as of the end of the data
3. keep the ranker's top 200, apply the popularity penalty (alpha = 0.3), keep the top 10
4. explain each one with the retriever that ranked it higher:
   - co-occurrence: "Players who liked X also liked this"
   - content:       "Because you liked X: shares tag1, tag2, tag3"

Then two extra rows of 5 (D32), each straight from one retriever's own order, so the page can show
what each retriever found on its own: "similar tags" (content) and "players also liked"
(co-occurrence). A game appears at most once on the page: each row skips games already shown above it.

Also `search_games` for finding a game by title, `top_games` for the home page's most-liked list,
and `load_artifacts` to read data/serve etc.
"""

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from steam_rec.candidates import TOP_N, keep_top_n
from steam_rec.content import CONTENT_DIR
from steam_rec.ranker import FEATURES, MISSING_RANK, MODEL_PATH, build_candidates
from steam_rec.rerank import popularity_penalty
from steam_rec.serve_build import SERVE_DIR

ALPHA = 0.3
N_RECOMMENDATIONS = 10
ROW_SIZE = 5
N_TOP_GAMES = 30
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
    """Read the serving files and the ranker model from disk, using as little memory as possible (D40)."""
    game_features = pd.read_parquet(serve_dir / "game_features.parquet")
    # Only the 3 columns scoring uses (recommend_from_neighbors).
    neighbor_columns = ["app_id", "neighbor_id", "similarity"]
    cooc_neighbors = pd.read_parquet(serve_dir / "cooc_neighbors.parquet", columns=neighbor_columns)
    # 5 million rows: app_ids fit easily in 32 bits, which halves the two id columns (101 MB -> 61 MB).
    content_neighbors = pd.read_parquet(content_dir / "neighbors.parquet", columns=neighbor_columns)
    content_neighbors["app_id"] = content_neighbors["app_id"].astype("int32")
    content_neighbors["neighbor_id"] = content_neighbors["neighbor_id"].astype("int32")
    return Artifacts(
        games=pd.read_parquet(serve_dir / "games.parquet"),
        cooc_neighbors=cooc_neighbors,
        content_neighbors=content_neighbors,
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


def top_games(games: pd.DataFrame, game_features: pd.DataFrame, n: int = N_TOP_GAMES) -> pd.DataFrame:
    """The `n` most-liked real games of all time (likes in our reviews), for the home page (D41)."""
    is_software = []
    for tags in games["tags"]:
        is_software.append("Software" in tags)
    # Software (e.g. Wallpaper Engine) isn't caught by the non-game filter, but Steam tags it "Software".
    real_games = games[~games["is_non_game"] & ~pd.Series(is_software, index=games.index)]

    ranked = real_games.merge(game_features[["app_id", "log_likes"]], on="app_id")
    ranked = ranked.sort_values("log_likes", ascending=False)
    return ranked.head(n).reset_index(drop=True)


def explain_players_also_liked(row: pd.Series, games_by_id: pd.DataFrame) -> dict:
    """The co-occurrence reason: the seed whose players liked this game most."""
    seed_id = int(row["cooc_best_seed"])
    seed_title = games_by_id.loc[seed_id, "title"]
    return {"type": "players_also_liked", "seed_app_id": seed_id, "shared_tags": [],
            "text": f"Players who liked {seed_title} also liked this"}


def explain_similar_tags(row: pd.Series, games_by_id: pd.DataFrame) -> dict:
    """The content reason: the most similar seed and up to 3 tags the two games share."""
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


def explain(row: pd.Series, games_by_id: pd.DataFrame) -> dict:
    """The reason for one top-10 recommendation: the retriever that ranked it higher explains it."""
    # On a tie, co-occurrence.
    if row["cooc_rank"] <= row["content_rank"]:
        return explain_players_also_liked(row, games_by_id)
    return explain_similar_tags(row, games_by_id)


def game_card(app_id: int, games_by_id: pd.DataFrame, reason: dict) -> dict:
    """One recommended game as the API returns it: id, title, year, price and the reason."""
    game = games_by_id.loc[app_id]
    return {
        "app_id": int(app_id),
        "title": game["title"],
        "release_year": int(game["release_year"]),
        "price": float(game["price_final"]),
        "reason": reason,
    }


def retriever_row(candidates: pd.DataFrame, rank_column: str, shown_ids: list[int], size: int) -> pd.DataFrame:
    """One retriever's best `size` candidates, in its own order, skipping games already shown."""
    # MISSING_RANK marks candidates this retriever didn't find (only the other one did).
    found = candidates[candidates[rank_column] < MISSING_RANK]
    found = found[~found["app_id"].isin(shown_ids)]
    return found.sort_values(rank_column).head(size)


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
    seed_ids: list[int],
    artifacts: Artifacts,
    n: int = N_RECOMMENDATIONS,
    alpha: float = ALPHA,
    row_size: int = ROW_SIZE,
) -> dict:
    """Recommend games for someone who liked `seed_ids`: the top `n` plus one row per retriever.

    Returns {"recommendations": [...], "similar_tags": [...], "players_also_liked": [...]},
    each a list of games with a reason.
    """
    games_by_id = artifacts.games.set_index("app_id")
    unknown = []
    for app_id in seed_ids:
        if app_id not in games_by_id.index:
            unknown.append(app_id)
    if unknown:
        raise ValueError(f"unknown app_ids: {unknown}")

    result = {"recommendations": [], "similar_tags": [], "players_also_liked": []}

    # One "user" (id 0) with these seeds. Only the seeds' neighbour rows are needed.
    seeds = pd.DataFrame({"user_id": 0, "app_id": seed_ids})
    cooc = artifacts.cooc_neighbors[artifacts.cooc_neighbors["app_id"].isin(seed_ids)]
    content = artifacts.content_neighbors[artifacts.content_neighbors["app_id"].isin(seed_ids)]
    candidates = build_candidates(seeds, cooc, content)

    # Never recommend soundtracks / DLC (D10), whichever retriever found them.
    non_game_ids = artifacts.games.loc[artifacts.games["is_non_game"], "app_id"]
    candidates = candidates[~candidates["app_id"].isin(non_game_ids)]
    if len(candidates) == 0:
        return result

    # The top n: ranker + popularity penalty, exactly as evaluated.
    with_features = candidates.merge(artifacts.game_features, on="app_id", how="left")
    ranked, final = score_and_rerank(with_features, artifacts.model, artifacts.max_log_likes, alpha)
    final = final.head(n)
    # Bring back the candidate columns (ranks, best seeds) that the explanation needs.
    final = final[["app_id", "rank"]].merge(ranked.drop(columns=["score", "rank"]), on="app_id")
    for _, row in final.sort_values("rank").iterrows():
        result["recommendations"].append(game_card(row["app_id"], games_by_id, explain(row, games_by_id)))

    # The extra rows, each from one retriever. Every game shown so far is skipped.
    shown_ids = []
    for rec in result["recommendations"]:
        shown_ids.append(rec["app_id"])

    for _, row in retriever_row(candidates, "content_rank", shown_ids, row_size).iterrows():
        result["similar_tags"].append(game_card(row["app_id"], games_by_id, explain_similar_tags(row, games_by_id)))
        shown_ids.append(int(row["app_id"]))

    for _, row in retriever_row(candidates, "cooc_rank", shown_ids, row_size).iterrows():
        reason = explain_players_also_liked(row, games_by_id)
        result["players_also_liked"].append(game_card(row["app_id"], games_by_id, reason))
        shown_ids.append(int(row["app_id"]))

    return result
