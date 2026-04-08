FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────────
COPY . .

# ── Hugging Face Spaces expects port 7860 ────────────────────────────────────
EXPOSE 7860

# ── Non-root user (HF Spaces best practice) ──────────────────────────────────
RUN useradd -m -u 1000 hfuser && chown -R hfuser /app
USER hfuser

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
