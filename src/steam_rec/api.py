"""Web API for the recommender (decision D30).

Endpoints:
- GET  /                       -> the home page: the 30 most-liked games (static/home.html)
- GET  /recommendations        -> the page where you pick games and get recommendations (static/recommend.html)
- GET  /static/...             -> the pages' shared CSS and JavaScript
- GET  /health                 -> {"status": "ok"}, for Docker / hosting health checks
- GET  /search?q=hollow        -> games whose title contains the text, most-reviewed first
- GET  /top?limit=30           -> the most-liked games of all time, for the home page
- POST /recommend              -> body {"app_ids": [3 to 5 different games]}; the top 10 games, plus
                                  5 more from each retriever (similar tags, players also liked), with reasons

The model files are loaded once when the server starts, not on every request.
Interactive docs (try the endpoints in a browser): http://127.0.0.1:8000/docs
Run with: uv run uvicorn steam_rec.api:app --reload
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from steam_rec.recommender import N_TOP_GAMES, Artifacts, load_artifacts, recommend, search_games, top_games
from steam_rec.split import MAX_SEEDS, MIN_SEEDS

STATIC_DIR = Path(__file__).parent / "static"
HOME_PAGE = STATIC_DIR / "home.html"
RECOMMEND_PAGE = STATIC_DIR / "recommend.html"
N_TOP_TAGS = 3


class GameOut(BaseModel):
    """One search result."""

    app_id: int
    title: str
    release_year: int


class TopGame(BaseModel):
    """One game in the home page's most-liked list."""

    app_id: int
    title: str
    release_year: int
    price: float
    positive_ratio: int
    user_reviews: int
    tags: list[str]


class Reason(BaseModel):
    """Why a game was recommended."""

    type: str
    seed_app_id: int
    shared_tags: list[str]
    text: str


class Recommendation(BaseModel):
    """One recommended game with its reason."""

    app_id: int
    title: str
    release_year: int
    price: float
    reason: Reason


class RecommendRequest(BaseModel):
    """The games the person liked: 3 to 5 different app_ids, as during evaluation (D19)."""

    app_ids: list[int] = Field(min_length=MIN_SEEDS, max_length=MAX_SEEDS)

    @field_validator("app_ids")
    @classmethod
    def no_duplicates(cls, app_ids: list[int]) -> list[int]:
        """Reject the same game given twice."""
        if len(set(app_ids)) != len(app_ids):
            raise ValueError("app_ids must be different games")
        return app_ids


class RecommendResponse(BaseModel):
    """The top recommendations (best first), then one extra row per retriever (D32)."""

    recommendations: list[Recommendation]
    similar_tags: list[Recommendation]
    players_also_liked: list[Recommendation]


def create_app(load: Callable[[], Artifacts] = load_artifacts) -> FastAPI:
    """Build the app; `load` provides the artifacts (the real files, or a small fake set in tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup: everything before `yield`. Requests are served while it's paused there.
        app.state.artifacts = load()
        yield

    app = FastAPI(title="Steam game recommender", lifespan=lifespan)

    # The pages' shared styles.css and common.js, served as plain files.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """The home page; it calls /top from the browser."""
        return FileResponse(HOME_PAGE)

    @app.get("/recommendations", include_in_schema=False)
    def recommendations_page() -> FileResponse:
        """The recommendations page; it calls /search and /recommend from the browser."""
        return FileResponse(RECOMMEND_PAGE)

    @app.get("/health")
    def health() -> dict:
        """Answer quickly if the server is up and the artifacts are loaded."""
        return {"status": "ok"}

    @app.get("/search", response_model=list[GameOut])
    def search(q: str = Query(min_length=2), limit: int = Query(default=10, ge=1, le=25)) -> list[dict]:
        """Games whose title contains `q`, most-reviewed first."""
        found = search_games(q, app.state.artifacts.games, limit)
        results = []
        for game in found.itertuples():
            results.append({"app_id": int(game.app_id), "title": game.title, "release_year": int(game.release_year)})
        return results

    @app.get("/top", response_model=list[TopGame])
    def top(limit: int = Query(default=N_TOP_GAMES, ge=1, le=50)) -> list[dict]:
        """The most-liked games of all time, with their first few tags."""
        found = top_games(app.state.artifacts.games, app.state.artifacts.game_features, limit)
        results = []
        for game in found.itertuples():
            results.append({
                "app_id": int(game.app_id),
                "title": game.title,
                "release_year": int(game.release_year),
                "price": float(game.price_final),
                "positive_ratio": int(game.positive_ratio),
                "user_reviews": int(game.user_reviews),
                "tags": list(game.tags[:N_TOP_TAGS]),
            })
        return results

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend_games(request: RecommendRequest) -> dict:
        """The top 10 games for someone who liked `app_ids`, plus a row of 5 from each retriever."""
        try:
            result = recommend(request.app_ids, app.state.artifacts)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result

    return app


app = create_app()
