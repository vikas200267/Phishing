#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VERSION="$(cat models/CURRENT)"
echo "[1/7] Python compilation"
python -m compileall -q app scripts benchmark.py run_train.py tests
echo "[2/7] Test suite"
pytest -q --disable-warnings
echo "[3/7] Model integrity"
python - <<'PY'
from pathlib import Path
from app.registry import current, verify_model_integrity
v=current(); p=Path('models')/v/'model.joblib'
assert verify_model_integrity(v,p)
print(v, 'integrity=OK')
PY
echo "[4/7] Frozen regression"
python scripts/evaluate_frozen.py --version "$VERSION"
echo "[5/7] Real replay benchmark"
python benchmark.py --input data/realworld_load_corpus.json --batch 256
echo "[6/7] Dependency security"
if command -v pip-audit >/dev/null 2>&1; then
  pip-audit -r requirements.txt
else
  echo "pip-audit unavailable: CVE status UNVERIFIED"
fi


echo "[7/7] Phase 8–17 release gate (records external gates without fabricating success)"
python scripts/phase8_17_gate.py
