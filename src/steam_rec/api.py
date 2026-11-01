"""Web API for the recommender (decision D30).

Endpoints:
- GET  /                       -> the web page (static/index.html)
- GET  /health                 -> {"status": "ok"}, for Docker / hosting health checks
- GET  /search?q=hollow        -> games whose title contains the text, most-reviewed first
- POST /recommend              -> body {"app_ids": [3 to 5 different games]}; 10 games with reasons

The model files are loaded once when the server starts, not on every request.
Interactive docs (try the endpoints in a browser): http://127.0.0.1:8000/docs
Run with: uv run uvicorn steam_rec.api:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from steam_rec.recommender import Artifacts, load_artifacts, recommend, search_games
from steam_rec.split import MAX_SEEDS, MIN_SEEDS

INDEX_PAGE = Path(__file__).parent / "static" / "index.html"


class GameOut(BaseModel):
    """One search result."""

    app_id: int
    title: str
    release_year: int


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
    """The list of recommendations, best first."""

    recommendations: list[Recommendation]


def create_app(load: Callable[[], Artifacts] = load_artifacts) -> FastAPI:
    """Build the app; `load` provides the artifacts (the real files, or a small fake set in tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup: everything before `yield`. Requests are served while it's paused there.
        app.state.artifacts = load()
        yield

    app = FastAPI(title="Steam game recommender", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """The web page; it calls /search and /recommend from the browser."""
        return FileResponse(INDEX_PAGE)

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

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend_games(request: RecommendRequest) -> dict:
        """10 games for someone who liked `app_ids`, each with a reason."""
        try:
            recommendations = recommend(request.app_ids, app.state.artifacts)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        return {"recommendations": recommendations}

    return app


app = create_app()
