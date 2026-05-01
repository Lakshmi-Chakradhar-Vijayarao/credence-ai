FROM python:3.12-slim

WORKDIR /app

# System deps for Rust (optional, only needed to build the gate inside container)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[mcp]"

# Application code
COPY credence/ ./credence/
COPY evals/    ./evals/
COPY tests/    ./tests/
COPY quickstart.py ./

# Registry persists to a mounted volume in production
ENV CREDENCE_DB_PATH=/data/epistemic_registry.db
ENV CREDENCE_SESSION_DIR=/data/sessions
ENV CREDENCE_AUTO_PERSIST=1

VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "-m", "credence.mcp_server"]
