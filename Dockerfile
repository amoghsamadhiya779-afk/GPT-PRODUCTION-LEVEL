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

# Copy source code files
COPY app/ /app/app/
COPY model/ /app/model/
COPY training/ /app/training/
COPY configs/ /app/configs/

# Expose server ports
EXPOSE 7860
EXPOSE 8000
EXPOSE 8501

# Default runtime command (can be overridden in docker-compose)
CMD uvicorn app.api:app --host 0.0.0.0 --port $PORT
