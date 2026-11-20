from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import extract_url_features
from app.local_intel import DEFAULT_SOURCE_DIR, local_intel_metadata
from app.registry import current, verify_model_integrity
from app.url_model import load_bundle
from app.unicode_url import unicode_host_diagnostics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], timeout: int = 120) -> dict:
    try:
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            'command': ' '.join(cmd), 'returncode': r.returncode,
            'stdout': r.stdout[-3000:], 'stderr': r.stderr[-3000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {'command': ' '.join(cmd), 'returncode': 124,
                'stdout': (exc.stdout or '')[-3000:] if isinstance(exc.stdout, str) else '',
                'stderr': 'timed out'}


def main() -> int:
    checks = []
    checks.append(run([sys.executable, '-m', 'compileall', '-q', 'app', 'scripts', 'tests']))
    checks.append(run([sys.executable, '-m', 'pytest', '-q', '--disable-warnings'], timeout=180))
    checks.append(run([sys.executable, 'scripts/production_doctor.py']))

    version = current()
    model_path = ROOT / 'models' / str(version) / 'model.joblib'
    integrity_ok = False
    model_details = {}
    try:
        integrity_ok = bool(version and verify_model_integrity(version, model_path))
        model = load_bundle(model_path)
        sample_urls = [
            'https://www.example.org/account/login',
            'https://xn--pple-43d.example/login',
        ]
        probs = model.predict_proba(sample_urls)
        features = extract_url_features(sample_urls[1])
        raw_components = model.raw_components(sample_urls)
        unicode_info = unicode_host_diagnostics(sample_urls[1])
        model_details = {
            'version': version,
            'sha256': sha256(model_path),
            'feature_count': len(model.feature_names),
            'raw_component_count': int(raw_components.shape[1]),
            'raw_component_shape': list(raw_components.shape),
            'feature_extractor_count': len(features),
            'feature_names_match_extractor': set(model.feature_names) == set(features),
            'probabilities_in_unit_interval': bool(((probs >= 0) & (probs <= 1)).all()),
            'unicode_feature_detected': bool(features.get('has_punycode', 0)),
            'unicode_diagnostic_present': bool(unicode_info.get('punycode_present')),
            'fast_path_enabled': bool(getattr(model, 'fast_path_enabled', False)),
            'fast_path_disabled_safely_if_no_calibrator': (
                getattr(model, 'fast_calibrator', None) is not None
                or not bool(getattr(model, 'fast_path_enabled', False))
            ),
            'threshold': float(model.threshold),
            'suspicious_threshold': getattr(model, 'suspicious_threshold', None),
        }
    except Exception as exc:
        model_details = {'version': version, 'error': f'{type(exc).__name__}: {exc}'}

    model_checks = {
        'current_model_set': bool(version),
        'model_integrity': integrity_ok,
        'seven_component_model': model_details.get('raw_component_count') == 7,
        'feature_schema_matches_live_extractor': model_details.get('feature_names_match_extractor') is True,
        'probabilities_are_bounded': model_details.get('probabilities_in_unit_interval') is True,
        'unicode_features_present': model_details.get('unicode_feature_detected') is True and model_details.get('unicode_diagnostic_present') is True,
        'unsafe_fast_path_not_enabled': model_details.get('fast_path_disabled_safely_if_no_calibrator') is True,
    }

    leak_path = ROOT / 'data/manifests/leakage_audit.json'
    leak = run([
        sys.executable, 'scripts/audit_dataset_leakage.py',
        '--train', 'data/training/production_training.csv',
        '--test', 'data/realworld/archive/openphish_20261005_unseen.csv',
        '--out', str(leak_path.relative_to(ROOT)),
    ])
    checks.append(leak)
    try:
        leak_result = json.loads(leak_path.read_text(encoding='utf-8'))
    except Exception:
        leak_result = {'passed': False, 'parse_error': True}
    # Keep one full audit report. The historical phase-specific location is a
    # compact pointer so consumers can resolve it without storing a second copy.
    alias_path = ROOT / 'data/manifests/phase8_17_leakage_audit.json'
    alias_path.write_text(json.dumps({
        'canonical_report': 'data/manifests/leakage_audit.json',
        'sha256': sha256(leak_path) if leak_path.exists() else None,
        'passed': bool(leak_result.get('passed')),
    }, indent=2), encoding='utf-8')

    # Candidates are kept outside models/ and can never become active by accident.
    candidates = []
    candidate_root = ROOT / 'artifacts' / 'phase8_17' / 'candidate_evaluations'
    for metadata_path in sorted(candidate_root.glob('*/metadata.json')):
        version_name = metadata_path.parent.name
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metrics_path = metadata_path.parent / 'metrics.json'
            frozen_path = metadata_path.parent / 'frozen_regression_metrics.json'
            metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
            frozen = json.loads(frozen_path.read_text(encoding='utf-8')) if frozen_path.exists() else {}
            candidates.append({
                'version': version_name,
                'promotion_eligible': bool(metadata.get('promotion_eligible')),
                'release_status': metadata.get('release_status'),
                'outer': {k: metrics.get(k) for k in ('accuracy', 'precision', 'recall', 'f1')},
                'frozen': {k: frozen.get(k) for k in ('accuracy', 'precision', 'recall', 'f1')},
                'fast_path_enabled': metadata.get('fast_path', {}).get('enabled'),
                'active_artifact_directory_exists': (ROOT / 'models' / version_name).exists(),
            })
        except Exception as exc:
            candidates.append({'version': version_name, 'error': f'{type(exc).__name__}: {exc}'})

    archive_meta_path = ROOT / 'data/realworld/archive/openphish_20261005_unseen.csv.meta.json'
    archive_meta = {}
    if archive_meta_path.exists():
        try:
            archive_meta = json.loads(archive_meta_path.read_text(encoding='utf-8'))
        except Exception:
            archive_meta = {'parse_error': True}
    feed_paths = [DEFAULT_SOURCE_DIR / 'openphish.csv', DEFAULT_SOURCE_DIR / 'urlhaus.csv']
    feed_states = []
    for path in feed_paths:
        sidecar = path.with_suffix(path.suffix + '.meta.json')
        meta = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding='utf-8'))
            except Exception:
                meta = {'parse_error': True}
        feed_states.append({
            'path': str(path.relative_to(ROOT)), 'exists': path.exists(),
            'metadata_exists': sidecar.exists(), 'metadata': meta,
        })

    url_temporal_gate_path = ROOT / 'data/manifests/url_temporal_gate.json'
    try:
        url_temporal_gate = json.loads(url_temporal_gate_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        url_temporal_gate = {}

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'phase_range': '8-17',
        'release_model_version': version,
        'required_checks_passed': (
            all(x.get('returncode') == 0 for x in checks)
            and all(model_checks.values())
            and bool(leak_result.get('passed'))
        ),
        'checks': checks,
        'model_checks': model_checks,
        'model': model_details,
        'leakage_audit': leak_result,
        'challengers': candidates,
        'historic_positive_feed': {
            'path': 'data/realworld/archive/openphish_20261005_unseen.csv',
            'retrieved_at': archive_meta.get('accessed_at'),
            'status': 'historical_archive_not_current',
            'listed_urls_visited': False,
        },
        'live_feed_refresh': {
            'status': 'not_verified_in_this_runtime' if not any(x['exists'] and x['metadata_exists'] for x in feed_states) else 'local_feed_files_present_freshness_must_be_checked',
            'feeds': feed_states,
            'local_index_runtime_metadata': local_intel_metadata(),
        },
        'temporal_validation': {
            'status': 'BENCHMARK_SPLIT_PASS_PRODUCTION_MODEL_NOT_READY' if url_temporal_gate.get('status') == 'pass' else 'NOT_READY',
            'benchmark_dataset_gate': {
                'status': url_temporal_gate.get('status', 'not_ready'),
                'dataset': url_temporal_gate.get('source_dataset'),
                'cutoff': url_temporal_gate.get('cutoff'),
                'train_rows': url_temporal_gate.get('train_rows'),
                'test_rows': url_temporal_gate.get('test_rows'),
                'timestamp_coverage': url_temporal_gate.get('timestamp_coverage'),
                'exact_url_overlap': url_temporal_gate.get('exact_url_overlap'),
                'license': url_temporal_gate.get('license'),
            },
            'production_model_status': 'NOT_READY',
            'challenger_evaluation': (json.loads((ROOT / 'data/realworld/evaluations/phishvn_temporal/model_evaluation.json').read_text(encoding='utf-8')) if (ROOT / 'data/realworld/evaluations/phishvn_temporal/model_evaluation.json').exists() else None),
            'reason': 'PhishVN v5 supplies a CC BY 4.0 source-attested temporal split and a pre-cutoff challenger was evaluated. The challenger fails frozen regression, so it was not promoted. CURRENT has no authentic per-row training timestamps and is not temporally qualified.',
        },
        'cloud_scale': {
            'status': 'NOT_TESTED',
            'reason': 'A distributed cluster/load-generator run is required; sandbox replay is not a capacity result.',
            'target_rps': {'10m_in_15s': 10_000_000 / 15, '10m_in_10s': 1_000_000},
        },
        'dependency_cve_status': 'UNVERIFIED until scanners run against the resolved release in network-enabled CI',
    }
    out = ROOT / 'data/manifests/phase8_17_gate.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['required_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
