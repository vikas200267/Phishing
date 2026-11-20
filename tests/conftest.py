import os
from pathlib import Path
import sys

# Test-time secrets must be configured before importing app modules because
# app.config.settings is intentionally immutable after import.
os.environ.setdefault('PHISHGUARD_API_KEY', 'test-inference-key')
os.environ.setdefault('PHISHGUARD_ADMIN_KEY', 'test-admin-key')
os.environ.setdefault('AUTH_REQUIRED', 'true')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
