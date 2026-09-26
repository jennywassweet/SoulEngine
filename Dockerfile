FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app

# Use the image's Python, never a downloaded one; copy rather than hardlink
# from uv's cache (the cache is not on the same filesystem in a build).
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Dependencies from uv.lock, in their own layer: rebuilt only when the lock
# changes, not on every code edit.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application code
COPY engine/ ./engine/
COPY soul/ ./soul/
COPY config/ ./config/
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Run the engine
CMD ["python", "-m", "engine.main"]
