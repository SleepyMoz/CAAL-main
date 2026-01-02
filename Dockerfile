# syntax=docker/dockerfile:1

# CAAL - Voice Agent
# Lightweight Python agent for voice orchestration (GPU handled by Speaches)

# ============================================================================
# Base image - slim Python (agent doesn't need GPU, Speaches handles that)
# ============================================================================
FROM python:3.11-slim-bookworm AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ============================================================================
# Dependencies stage
# ============================================================================
FROM base AS deps

WORKDIR /app

# Copy files needed for dependency installation
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies (non-editable)
RUN uv sync --frozen --no-dev --no-editable

# ============================================================================
# Production image
# ============================================================================
FROM base AS runner

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash agent

# Copy virtual environment from deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy application code
COPY --chown=agent:agent src/ ./src/
COPY --chown=agent:agent voice_agent.py ./
COPY --chown=agent:agent prompt/ ./prompt/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Pre-download Silero VAD model (optional, reduces first-run delay)
# RUN python -c "from livekit.plugins import silero; silero.VAD.load()"

# Switch to non-root user
USER agent

# Default command - start mode for production
# Override with 'dev' for development
CMD ["python", "voice_agent.py", "start"]
