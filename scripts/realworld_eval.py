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
    h = host(u)
    parts = [x for x in h.split('.') if x]
    if len(parts) <= 2:
        return h
    suffix2 = '.'.join(parts[-2:])
    if suffix2 in {'co.uk','org.uk','ac.uk','gov.uk','com.au','net.au','org.au','co.in','com.br','co.jp','co.nz','co.za'}:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])


def load_training_domains(model_version: str) -> set[str]:
    path = ROOT / 'models' / model_version / 'training_domains.txt'
    if not path.exists():
        raise RuntimeError(f'{path} is missing; model cannot make a leakage-controlled live evaluation')
    return {line.strip().lower() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


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
    return {
        'support': total,
        'accuracy': (tp + tn) / max(1, total),
        'precision': tp / max(1, tp + fp),
        'recall': tp / max(1, tp + fn),
        'f1': 2 * tp / max(1, 2 * tp + fp + fn),
        'false_positive_rate': fp / max(1, fp + tn),
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
    args = ap.parse_args()

    from app.registry import current
    from app.url_model import load_bundle

    version = args.model_version or current()
    model = load_bundle(ROOT / 'models' / version / 'model.joblib')
    training_domains = load_training_domains(version)
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
        positives = [u for u in df['url'].dropna().astype(str).tolist() if registrable(u) not in training_domains]
        sources.append({'name':'captured-realworld-positive','path':str(cp),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'entries_seen':len(df),'unseen_domains_available':len(positives),'listed_urls_visited':False})
    else:
        for name, endpoint in [('OpenPhish', OPENPHISH), ('OpenPhish mirror', OPENPHISH_MIRROR)]:
            try:
                raw, resolved = fetch(endpoint, args.timeout)
                extracted = urls_from_bytes(raw)
                unseen = [u for u in extracted if registrable(u) not in training_domains]
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
                if re.match(r'^https?://[^\s]+$', u, re.I) and registrable(u) not in training_domains:
                    uh.append(u)
            positives.extend(uh)
            urlhaus_info = {'enabled': True, 'url': resolved, 'sha256': hashlib.sha256(raw).hexdigest(), 'entries_seen': len(uh)}
        except Exception as exc:
            urlhaus_info = {'enabled': True, 'error': f'{type(exc).__name__}: {exc}'}

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
        supplied=[u for u,l in zip(bdf['url'].astype(str),bdf['label']) if int(float(l))==0 and registrable(u) not in training_domains]
        benign_rows, benign_c = evaluate(model, list(dict.fromkeys(supplied))[:args.limit_benign], 0)
        benign_role='independently_verified_captured_labels'
        sources.append({'name':'captured-realworld-benign','path':str(bd),'sha256':hashlib.sha256(bd.read_bytes()).hexdigest(),'entries_seen':len(bdf),'unseen_domains_available':len(supplied),'role':'independently_verified'})
    elif args.benign_label_csv:
        bd = Path(args.benign_label_csv)
        with bd.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            supplied = [r['url'].strip() for r in reader if (r.get('label') or '').strip() == '0' and r.get('url')]
        supplied = [u for u in supplied if registrable(u) not in training_domains]
        benign_rows, benign_c = evaluate(model, list(dict.fromkeys(supplied))[:args.limit_benign], 0)
        benign_role = 'independently_verified_operator_labels'
    elif args.limit_benign <= 0:
        benign_role = 'not_requested'
    else:
        tr_raw, tr_resolved = fetch(TRANCO, args.timeout)
        tr_urls = tranco_urls(tr_raw)
        benign = [u for u in tr_urls if registrable(u) not in training_domains and norm(u) not in {norm(x) for x in positive}][:args.limit_benign]
        benign_rows, benign_c = evaluate(model, benign, 0)
        benign_role = 'Tranco popularity proxy; not a ground-truth benign label'
        sources.append({'name':'Tranco latest','url':tr_resolved,'sha256':hashlib.sha256(tr_raw).hexdigest(),'bytes':len(tr_raw),'entries_seen':len(tr_urls),'unseen_domains_available':len(benign),'role':'benign-proxy'})

    combined = Counter(pos_c); combined.update(benign_c)
    result = {
        'generated_at': now,
        'model_version': version,
        'threshold': float(model.threshold),
        'training_domain_count': len(training_domains),
        'positive_sample': metric_summary(pos_c),
        'benign_sample': metric_summary(benign_c),
        'benign_label_role': benign_role,
        'combined_metrics_only_if_benign_labels_are_ground_truth': metric_summary(dict(combined)) if benign_role == 'independently_verified_operator_labels' else None,
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
