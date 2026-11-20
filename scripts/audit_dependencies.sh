#!/usr/bin/env bash
set -euo pipefail

# Audits the declared runtime lock without mutating the environment.
# Run this inside a clean, network-enabled CI/build environment.
if ! python -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("pip_audit") else 1)'; then
  echo 'pip-audit is not installed. Install requirements-audit.txt in the CI environment.' >&2
  exit 2
fi
python -m pip_audit -r requirements.lock
