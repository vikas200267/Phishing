from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.domain import registrable_domain
from app.policy import valid_http_url
OPENPHISH = 'https://www.openphish.com/feed.txt'
OPENPHISH_MIRROR = 'https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt'
URLHAUS_TEMPLATE = 'https://urlhaus-api.abuse.ch/v2/files/exports/{key}/recent.csv'
TRANCO = 'https://tranco-list.eu/top-1m.csv.zip'


def fetch(url: str, timeout: int = 60, headers: dict[str, str] | None = None) -> tuple[bytes, str]:
    req = Request(url, headers=headers or {'User-Agent': 'PhishGuard-RealWorld-Evaluator/2.0'})
    with urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl()


def urls_from_bytes(data: bytes) -> list[str]:
    out, seen = [], set()
    for line in data.decode('utf-8', 'replace').splitlines():
        u = line.strip()
        if u and not u.startswith('#') and re.match(r'^https?://[^\s]+$', u, re.I) and u not in seen:
            seen.add(u); out.append(u)
    return out


def tranco_urls(data: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names:
            raise RuntimeError('Tranco ZIP contains no CSV')
        raw = z.read(names[0]).decode('utf-8', 'replace')
    out, seen = [], set()
    for row in csv.reader(io.StringIO(raw)):
        if not row:
            continue
        domain = row[-1].strip().lower()
        if domain and domain != 'domain':
            u = 'https://' + domain
            if u not in seen:
                seen.add(u); out.append(u)
    return out


def norm(u: str) -> str:
    return str(u).strip().lower().rstrip('/')


def host(u: str) -> str:
    try:
        return (urlsplit(norm(u)).hostname or '').lower().strip('.')
    except ValueError:
        return ''


def registrable(u: str) -> str:
    return registrable_domain(u)


def overlaps_training_domain(u: str, training_domains: set[str]) -> bool:
    """Conservatively support both legacy and current model domain lists."""
    try:
        h = host(u).encode('idna').decode('ascii')
    except UnicodeError:
        return True
    if not h:
        return True
    labels = h.split('.')
    # A legacy model might have recorded a broader suffix such as github.io.
    # Exclude every subdomain under it instead of claiming it is unseen.
    return any('.'.join(labels[i:]) in training_domains for i in range(len(labels)))


def load_training_domains(model_version: str) -> set[str]:
    path = ROOT / 'models' / model_version / 'training_domains.txt'
    if not path.exists():
        raise RuntimeError(f'{path} is missing; model cannot make a leakage-controlled live evaluation')
    return {line.strip().lower().encode('idna').decode('ascii') for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def load_input_domains(model_version: str) -> set[str]:
    """Reserve every recorded input domain, including the model's outer split."""
    import pandas as pd
    from app.train import url_series
    model_dir = ROOT / 'models' / model_version
    metadata = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
    sources = metadata.get('data_sources') or {}
    parameters = metadata.get('training_parameters') or {}
    domains = set()
    for name in ('phishing', 'legitimate', 'extra'):
        entry = sources.get(name)
        path_text = entry.get('path') if isinstance(entry, dict) else parameters.get(name)
        expected = entry.get('sha256') if isinstance(entry, dict) else entry
        if not path_text:
            if name != 'extra':
                raise RuntimeError(f'{name} training source is not recorded')
            continue
        path = Path(str(path_text).replace('\\', '/'))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'{name} training source missing or changed; cannot establish input disjointness')
        frame = pd.read_csv(path, low_memory=False)
        domains.update(registrable_domain(u) for u in url_series(frame).dropna().astype(str))
    domains.discard('')
    if not domains:
        raise RuntimeError('Recorded inputs contain no usable domain groups')
    return domains


def evaluate(model, urls: list[str], label: int, chunk: int = 512):
    tp = fp = fn = tn = 0
    rows = []
    for i in range(0, len(urls), chunk):
        batch = urls[i:i + chunk]
        probs = model.predict_proba(batch)
        for u, p in zip(batch, probs):
            pred = int(float(p) >= float(model.threshold))
            if label == 1:
                tp += pred; fn += 1 - pred
            else:
                fp += pred; tn += 1 - pred
            rows.append({'url': u, 'label': label, 'score': float(p), 'prediction': pred})
    return rows, {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def metric_summary(c):
    tp, fp, fn, tn = [int(c[k]) for k in ('tp','fp','fn','tn')]
    total = tp + fp + fn + tn
    has_both_classes = (tp + fn) > 0 and (tn + fp) > 0
    return {
        'support': total,
        'accuracy': (tp + tn) / total if has_both_classes else None,
        'precision': tp / (tp + fp) if has_both_classes and tp + fp else None,
        'recall': tp / (tp + fn) if tp + fn else None,
        'f1': 2 * tp / (2 * tp + fp + fn) if has_both_classes and 2 * tp + fp + fn else None,
        'false_positive_rate': fp / (fp + tn) if fp + tn else None,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def main():
    ap = argparse.ArgumentParser(description='Leakage-controlled live threat-feed evaluation. Listed URLs are never visited.')
    ap.add_argument('--limit-positive', type=int, default=1000)
    ap.add_argument('--limit-benign', type=int, default=1000)
    ap.add_argument('--out', default='artifacts/realworld')
    ap.add_argument('--timeout', type=int, default=60)
    ap.add_argument('--urlhaus-key', default='')
    ap.add_argument('--benign-label-csv', default='', help='Optional independently verified benign URL CSV with url,label=0. Without this, Tranco is reported only as a proxy.')
    ap.add_argument('--captured-positive-csv', default='', help='Replay a previously captured real threat-feed CSV instead of downloading live feeds. No listed URL is visited.')
    ap.add_argument('--captured-benign-csv', default='', help='Replay a previously captured independently verified benign CSV instead of downloading Tranco. label=0 required.')
    ap.add_argument('--model-version', default='', help='Evaluate a specific registered or candidate model; defaults to models/CURRENT.')
    ap.add_argument('--allow-training-domains-only', action='store_true', help='Permit a weaker legacy replay when original hashed input files are unavailable.')
    args = ap.parse_args()

    from app.registry import current
    from app.url_model import load_bundle

    version = args.model_version or current()
    model = load_bundle(ROOT / 'models' / version / 'model.joblib')
    training_domains = load_training_domains(version)
    input_exclusion = 'all_recorded_inputs'
    try:
        training_domains |= load_input_domains(version)
    except (OSError, ValueError, RuntimeError) as exc:
        if not args.allow_training_domains_only:
            raise SystemExit(str(exc)) from exc
        input_exclusion = 'training_domains_only; recorded inputs unavailable or changed'
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    sources = []

    positives = []
    if args.captured_positive_csv:
        cp = Path(args.captured_positive_csv)
        if not cp.exists():
            raise SystemExit(f'Captured positive CSV not found: {cp}')
        df = __import__('pandas').read_csv(cp)
        if 'url' not in df.columns:
            raise SystemExit('Captured positive CSV must contain a url column')
        source_urls = df['url'].dropna().astype(str).tolist()
        positives = [u for u in source_urls if valid_http_url(u) and not overlaps_training_domain(u, training_domains)]
        sources.append({'name':'captured-realworld-positive','path':str(cp),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'entries_seen':len(df),'invalid_http_urls':sum(not valid_http_url(u) for u in source_urls),'unseen_domains_available':len(positives),'listed_urls_visited':False})
    else:
        for name, endpoint in [('OpenPhish', OPENPHISH), ('OpenPhish mirror', OPENPHISH_MIRROR)]:
            try:
                raw, resolved = fetch(endpoint, args.timeout)
                extracted = urls_from_bytes(raw)
                unseen = [u for u in extracted if valid_http_url(u) and not overlaps_training_domain(u, training_domains)]
                positives.extend(unseen)
                sources.append({'name': name, 'url': resolved, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'entries_seen': len(extracted), 'unseen_domains_available': len(unseen)})
                if unseen:
                    break
            except Exception as exc:
                sources.append({'name': name, 'url': endpoint, 'error': f'{type(exc).__name__}: {exc}'})

    urlhaus_info = {'enabled': bool(args.urlhaus_key)}
    if args.urlhaus_key:
        endpoint = URLHAUS_TEMPLATE.format(key=args.urlhaus_key)
        try:
            raw, resolved = fetch(endpoint, args.timeout)
            text = raw.decode('utf-8', 'replace')
            reader = csv.DictReader([x for x in text.splitlines() if x and not x.startswith('#')])
            uh = []
            for row in reader:
                u = (row.get('url') or row.get('URL') or '').strip()
                if valid_http_url(u) and not overlaps_training_domain(u, training_domains):
                    uh.append(u)
            positives.extend(uh)
            parsed = urlsplit(resolved)
            urlhaus_info = {'enabled': True, 'source': f'{parsed.scheme}://{parsed.hostname}', 'sha256': hashlib.sha256(raw).hexdigest(), 'entries_seen': len(uh)}
        except Exception as exc:
            # Provider exceptions can include the authenticated URL.
            urlhaus_info = {'enabled': True, 'error_type': type(exc).__name__}

    positive = list(dict.fromkeys(positives))[:args.limit_positive]
    if not positive:
        raise SystemExit('No unseen-domain real-world positive URLs were obtained; refusing to produce a synthetic result.')

    pos_rows, pos_c = evaluate(model, positive, 1)

    # Optional strict benign labels supplied by the operator; otherwise use a
    # clearly labeled Tranco popularity proxy and never call it ground truth.
    benign_rows = []; benign_c = {'tp':0,'fp':0,'fn':0,'tn':0}; benign_role = 'none'
    if args.captured_benign_csv:
        bd = Path(args.captured_benign_csv)
        if not bd.exists():
            raise SystemExit(f'Captured benign CSV not found: {bd}')
        bdf=__import__('pandas').read_csv(bd)
        if 'url' not in bdf.columns or 'label' not in bdf.columns:
            raise SystemExit('Captured benign CSV must contain url and label columns')
        supplied=[u for u,l in zip(bdf['url'].astype(str),bdf['label']) if int(float(l))==0 and valid_http_url(u) and not overlaps_training_domain(u, training_domains)]
        benign_rows, benign_c = evaluate(model, list(dict.fromkeys(supplied))[:args.limit_benign], 0)
        benign_role='independently_verified_captured_labels'
        sources.append({'name':'captured-realworld-benign','path':str(bd),'sha256':hashlib.sha256(bd.read_bytes()).hexdigest(),'entries_seen':len(bdf),'unseen_domains_available':len(supplied),'role':'independently_verified'})
    elif args.benign_label_csv:
        bd = Path(args.benign_label_csv)
        with bd.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            supplied = [r['url'].strip() for r in reader if (r.get('label') or '').strip() == '0' and r.get('url')]
        supplied = [u for u in supplied if valid_http_url(u) and not overlaps_training_domain(u, training_domains)]
        benign_rows, benign_c = evaluate(model, list(dict.fromkeys(supplied))[:args.limit_benign], 0)
        benign_role = 'independently_verified_operator_labels'
    elif args.limit_benign <= 0:
        benign_role = 'not_requested'
    else:
        tr_raw, tr_resolved = fetch(TRANCO, args.timeout)
        tr_urls = tranco_urls(tr_raw)
        benign = [u for u in tr_urls if valid_http_url(u) and not overlaps_training_domain(u, training_domains) and norm(u) not in {norm(x) for x in positive}][:args.limit_benign]
        benign_rows, benign_c = evaluate(model, benign, 0)
        benign_role = 'Tranco popularity proxy; not a ground-truth benign label'
        sources.append({'name':'Tranco latest','url':tr_resolved,'sha256':hashlib.sha256(tr_raw).hexdigest(),'bytes':len(tr_raw),'entries_seen':len(tr_urls),'unseen_domains_available':len(benign),'role':'benign-proxy'})

    combined = Counter(pos_c); combined.update(benign_c)
    result = {
        'generated_at': now,
        'model_version': version,
        'threshold': float(model.threshold),
        'training_domain_count': len(training_domains),
        'input_domain_exclusion': input_exclusion,
        'positive_sample': metric_summary(pos_c),
        'benign_sample': metric_summary(benign_c),
        'benign_label_role': benign_role,
        'combined_metrics_only_if_benign_labels_are_ground_truth': metric_summary(dict(combined)) if benign_role in {'independently_verified_operator_labels', 'independently_verified_captured_labels'} else None,
        'urlhaus': urlhaus_info,
        'sources': sources,
        'visited_listed_urls': False,
        'method': 'Captured/live feed replay after exact normalized URL and registrable-domain exclusion against the model training domains.' if (args.captured_positive_csv or args.captured_benign_csv) else 'Live feed replay after exact normalized URL and registrable-domain exclusion against the model training domains.',
        'limitations': [
            'OpenPhish/URLhaus metrics are positive-class recall only.',
            'Tranco is a popularity ranking, not a benign ground-truth label.',
            'Strict real-world precision requires independently verified benign labels or a reputation-confirmed evaluation population.',
        ],
    }
    (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['url','label','score','prediction']); w.writeheader(); w.writerows(pos_rows + benign_rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
