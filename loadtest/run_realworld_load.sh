#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python "$ROOT/scripts/build_realworld_corpus.py" --out "$ROOT/data/live_load_corpus.json"
: "${PHISHGUARD_API_KEY:?set PHISHGUARD_API_KEY}"
: "${PHISHGUARD_ADMIN_KEY:?set PHISHGUARD_ADMIN_KEY}"
API_REPLICAS="${API_REPLICAS:-16}"
LOAD_GENERATORS="${LOAD_GENERATORS:-8}"
TARGET_RPS_PER_GENERATOR="${TARGET_RPS_PER_GENERATOR:-125000}"
export TARGET_RPS="$TARGET_RPS_PER_GENERATOR"
export DURATION="${DURATION:-60s}"
export MODE="${MODE:-inference}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
docker compose -f "$ROOT/loadtest/docker-compose.loadtest.yml" up --build --scale api="$API_REPLICAS" --scale loadgen="$LOAD_GENERATORS"
