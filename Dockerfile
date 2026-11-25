FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_CONCURRENCY=1 \
    INFERENCE_THREADS=1 \
    INFERENCE_EXECUTOR_WORKERS=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
RUN groupadd --system --gid 10001 phishguard && useradd --system --uid 10001 --gid 10001 --create-home phishguard

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY app ./app
COPY models ./models
COPY data/patterns.json ./data/patterns.json
COPY data/realworld/trusted_homepages.csv ./data/realworld/trusted_homepages.csv
COPY data/realworld/legitimate_challenge.csv ./data/realworld/legitimate_challenge.csv
COPY scripts/refresh_feeds.py ./scripts/refresh_feeds.py
COPY scripts/build_realworld_corpus.py ./scripts/build_realworld_corpus.py

RUN mkdir -p /app/live_feeds /app/runtime \
    && chown -R phishguard:phishguard /app
USER phishguard

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).read()"

# One Uvicorn process per container. Scale containers horizontally.
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","8000","--workers","1","--no-access-log"]
