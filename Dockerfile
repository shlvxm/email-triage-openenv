FROM python:3.11-slim

# Enable CLI mode for OpenEnv validation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLI_MODE=true \
    TASK_LEVEL=medium \
    MAX_STEPS=50

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Run in CLI mode - outputs [START]/[STEP]/[END] blocks to stdout
CMD ["python", "inference.py"]
