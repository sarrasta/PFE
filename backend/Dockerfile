# Tunisie Telecom Retention Analytics — Backend (Flask + ML, CPU-only)
#
# No GPU/CUDA is required: the deployed models (SVC, RandomForest, KMeans,
# Ridge, LogisticRegression, GradientBoosting — see RAPPORT_PFE_FINAL.md) are
# all classic scikit-learn estimators trained on a few hundred to ~10,000
# rows. See README "GPU requirements" for details.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# psycopg2-binary needs libpq at runtime; gcc only needed transiently to
# build a couple of sklearn/scipy wheels that don't ship manylinux binaries
# for every platform — removed again after pip install to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# A single worker process on purpose: the trained models and scored client
# table live in that process's memory (see app/ml/pipeline.py), so running
# more than one worker would train everything twice and split traffic across
# two inconsistent caches. Threads still give real concurrency for an
# internal 5-user tool; scale further with a shared cache (Redis) first if
# this is ever fronted by many more users.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "run:app"]
