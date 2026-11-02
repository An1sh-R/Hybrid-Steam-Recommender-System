# Container image for the recommender web app (decision D34).
# Contains the code, the runtime dependencies and the ~50MB of serving files; never the raw data.
# The serving files must exist locally first (run `uv run dvc pull`, or `uv run dvc repro`).
#
# Build: docker build -t steam-rec .
# Run:   docker run --rm -p 8000:8000 steam-rec     then open http://127.0.0.1:8000
# Fits in 512MB of RAM (Render's free tier, D40): ~340MB idle, ~360MB after many requests.

FROM python:3.12-slim-bookworm

# LightGBM needs the OpenMP runtime library, which the slim image leaves out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv, copied from its official image, pinned to the version used in development.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

# Use the image's own Python (never download one), compile .py files at build time for a
# faster start, and copy packages instead of linking them.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Step 1: dependencies only. Docker caches this layer until pyproject.toml or uv.lock change,
# so editing the code doesn't reinstall pandas, LightGBM, ...
# The cache mount keeps uv's download cache (~430MB) outside the image: reused between builds,
# never shipped.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Step 2: the code, then the project itself. Installed "editable" (uv's default), so the code
# stays in /app/src and PROJECT_ROOT in data.py points at /app, where the data files go.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Step 3: the serving files, at the same paths as in the project.
COPY data/serve ./data/serve
COPY data/content/neighbors.parquet ./data/content/neighbors.parquet
COPY models/ranker.txt ./models/ranker.txt

# Run as an ordinary user, not root.
RUN useradd --create-home appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

# Memory settings (D40). Without them the app is killed at startup under a 512MB limit.
# - ARROW_DEFAULT_MEMORY_POOL=system: Arrow (which reads the parquet files) uses the normal C
#   allocator instead of its own pool, which keeps freed memory instead of giving it back.
# - MALLOC_ARENA_MAX=2: glibc's allocator keeps at most 2 memory pools instead of one per thread,
#   so less memory sits unused in half-empty pools.
ENV ARROW_DEFAULT_MEMORY_POOL=system \
    MALLOC_ARENA_MAX=2

# Hosts like Render say which port to listen on in $PORT; 8000 when it isn't set.
ENV PORT=8000
EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering. The slim image has no curl,
# so Python's own HTTP client does the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ['PORT']; urllib.request.urlopen('http://127.0.0.1:' + port + '/health', timeout=3)"

# `sh -c` so that $PORT is filled in when the container starts; `exec` makes uvicorn replace the
# shell, so it receives the stop signal directly and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn steam_rec.api:app --host 0.0.0.0 --port $PORT"]
