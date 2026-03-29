FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── App dir ───────────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── HF Spaces: non-root user required ────────────────────────────────────────
RUN useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Environment Variables ─────────────────────────────────────────────────────
# (Streamlit ones left for local execution, but Uvicorn uses port 7860)
ENV STREAMLIT_SERVER_PORT=8501
ENV PORT=7860

EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start the OpenEnv REST API server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
