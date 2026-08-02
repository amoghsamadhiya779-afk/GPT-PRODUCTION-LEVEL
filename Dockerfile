# Dockerfile
# ============================================================
#  GPT-2 Serving & Dashboard Docker Container
# ============================================================

FROM python:3.10-slim

# Set environment configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set workspace
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache tiktoken vocabulary to prevent runtime download hangs
ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken_cache
RUN mkdir -p /app/data/tiktoken_cache && python -c "import tiktoken; tiktoken.get_encoding('gpt2')"

# Copy source code files
COPY app/ /app/app/
COPY model/ /app/model/
COPY training/ /app/training/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY checkpoints/adapters/ /app/checkpoints/adapters/

# Run as a non-root user. /app/data and /app/checkpoints/adapters are written
# to at runtime (feedback, Teach Mode adapters, tiktoken cache), so they need
# to be owned by the user the process actually runs as.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/checkpoints/adapters /app/data \
    && chown -R appuser:appuser /app
USER appuser

# Expose server ports
EXPOSE 7860
EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default runtime command (can be overridden in docker-compose)
CMD uvicorn app.api:app --host 0.0.0.0 --port $PORT
