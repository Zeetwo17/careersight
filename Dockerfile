FROM python:3.11-slim

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2

WORKDIR /app

# ── Dependencies (cached layer) ─────────────────────────────────────────────
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# ── Project files ────────────────────────────────────────────────────────────
COPY . /app

# ── HF Spaces expects port 7860 ─────────────────────────────────────────────
EXPOSE 7860

# ── Start ────────────────────────────────────────────────────────────────────
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
