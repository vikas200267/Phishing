from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
EXPECTED_SCHEMA = 6
REQUIRED_COMPONENTS = {
    'xgboost_structured', 'lightgbm_structured', 'extratrees_structured',
    'logistic_url_tfidf', 'logistic_host_tfidf', 'linear_svc_url_char',
    'logistic_url_charwb',
}


def model_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _validate_model_dir(version: str | Path) -> dict:
    p = MODELS / str(version)
    if not p.is_dir():
        raise RuntimeError(f'model directory missing: {p.name}')
    if not (p / 'COMPLETE').exists():
        raise RuntimeError(f'model is not marked COMPLETE: {p.name}')
    model_path = p / 'model.joblib'
    meta_path = p / 'metadata.json'
    features_path = p / 'features.json'
    if not model_path.exists() or not meta_path.exists() or not features_path.exists():
        raise RuntimeError(f'model package is incomplete: {p.name}')
    meta = _read_json(meta_path)
    features = _read_json(features_path)
    if meta.get('version') != p.name:
        raise RuntimeError(f'model metadata version mismatch: {p.name}')
    if meta.get('schema_version') != EXPECTED_SCHEMA:
        raise RuntimeError(f'unsupported schema for {p.name}: {meta.get("schema_version")}')
    if meta.get('promotion_eligible') is False and p.name == current():
        # A currently deployed artifact is allowed to remain current during a
        # controlled rollback/migration, but explicit promotion is blocked below.
        pass
    if int(meta.get('feature_count', 0)) != len(features):
        raise RuntimeError(f'feature-count mismatch for {p.name}')
    comps = set(meta.get('model_components') or meta.get('training_parameters', {}).get('component_models') or [])
    if comps and not REQUIRED_COMPONENTS.issubset(comps):
        raise RuntimeError(f'missing required model components for {p.name}')
    if not comps:
        # Older v7 metadata listed the components in the model field only. Allow
        # only the known 88-feature family if all structural artifacts are present.
        if len(features) != 88:
            raise RuntimeError(f'component metadata missing for {p.name}')
    return {'path': p, 'metadata': meta, 'features': features}


def update_manifest(version: str) -> str:
    info = _validate_model_dir(version)
    model_path = info['path'] / 'model.joblib'
    manifest_path = MODELS / 'MANIFEST_SHA256.json'
    data = {}
    if manifest_path.exists():
        try:
            existing = _read_json(manifest_path)
            if isinstance(existing, list):
                for x in existing:
                    if not isinstance(x, dict):
                        continue
                    key = x.get('key') or x.get('version')
                    digest = x.get('model_sha256') or x.get('sha256')
                    if key and digest:
                        data[str(key)] = str(digest)
            elif isinstance(existing, dict):
                for k, v in existing.items():
                    if isinstance(v, str):
                        data[str(k)] = v
        except Exception:
            data = {}
    data[version] = artifact_sha256(model_path)
    email_path = info['path'] / 'email_model.joblib'
    if email_path.exists():
        data[f'{version}:email_model'] = artifact_sha256(email_path)
    fd, tmp_name = tempfile.mkstemp(prefix='MANIFEST.', dir=str(MODELS), text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            payload = [{'key': k, 'sha256': data[k]} for k in sorted(data) if data[k]]
            # Backward-compatible model_sha256 key for URL model entries.
            for item in payload:
                if ':email_model' not in item['key']:
                    item['version'] = item['key']
                    item['model_sha256'] = item['sha256']
            json.dump(payload, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_name, manifest_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return data[version]


def verify_artifact_integrity(key: str, path: Path) -> bool:
    manifest_path = MODELS / 'MANIFEST_SHA256.json'
    if not manifest_path.exists():
        raise RuntimeError('model integrity manifest is missing')
    data = _read_json(manifest_path)
    if isinstance(data, list):
        mapped = {}
        for x in data:
            if not isinstance(x, dict):
                continue
            k = x.get('key') or x.get('version')
            digest = x.get('sha256') or x.get('model_sha256')
            if k and digest:
                mapped[str(k)] = str(digest)
        data = mapped
    expected = data.get(key)
    if not expected:
        raise RuntimeError(f'no integrity hash for artifact {key}')
    actual = artifact_sha256(path)
    if not hmac.compare_digest(actual, str(expected)):
        raise RuntimeError(f'artifact integrity check failed for {key}')
    return True


def verify_model_integrity(version: str, path: Path) -> bool:
    _validate_model_dir(version)
    return verify_artifact_integrity(version, path)


def current():
    p = MODELS / 'CURRENT'
    return p.read_text(encoding='utf-8').strip() if p.exists() else None


def _meta_compatible(p: Path) -> bool:
    try:
        info = _validate_model_dir(p.name)
        meta = info['metadata']
        if meta.get('promotion_eligible') is False:
            return False
        comps = set(meta.get('model_components') or meta.get('training_parameters', {}).get('component_models') or [])
        if comps:
            return REQUIRED_COMPONENTS.issubset(comps)
        return len(info['features']) == 88
    except Exception:
        return False


def versions():
    return sorted(
        [p.name for p in MODELS.iterdir() if p.is_dir() and p.name.startswith('v') and _meta_compatible(p)]
    ) if MODELS.exists() else []


def _atomic_pointer(version: str):
    target = MODELS / 'CURRENT'
    fd, tmp = tempfile.mkstemp(prefix='CURRENT.', dir=str(MODELS), text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(version + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def promote(version: str):
    info = _validate_model_dir(version)
    if info['metadata'].get('promotion_eligible') is False:
        raise RuntimeError(f'model {version} failed its recorded promotion gate')
    verify_model_integrity(version, MODELS / version / 'model.joblib')
    old = current()
    _atomic_pointer(version)
    return {'previous': old, 'current': version}


def rollback():
    vs = versions()
    cur = current()
    earlier = [v for v in vs if v != cur and v < cur]
    if not earlier:
        raise RuntimeError('No earlier compatible model available')
    return promote(earlier[-1])
