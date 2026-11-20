from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OPENPHISH = os.getenv('OPENPHISH_FEED_URL', 'https://www.openphish.com/feed.txt')
OPENPHISH_MIRROR = os.getenv('OPENPHISH_MIRROR_URL', 'https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt')
URLHAUS_TEMPLATE = os.getenv('URLHAUS_EXPORT_URL_TEMPLATE', 'https://urlhaus-api.abuse.ch/v2/files/exports/{key}/recent.csv')


def fetch(url: str, headers: dict[str, str], timeout: int = 60) -> bytes:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def extract_urls(raw: bytes) -> list[str]:
    out, seen = [], set()
    for line in raw.decode('utf-8', 'replace').splitlines():
        value = line.strip()
        if re.match(r'^https?://[^\s]+$', value, re.I) and value not in seen:
            seen.add(value); out.append(value)
    return out


def main():
    ap = argparse.ArgumentParser(description='Build a load corpus only from current real-world threat feeds; never generates test URLs.')
    ap.add_argument('--out', default='data/live_load_corpus.json')
    ap.add_argument('--limit', type=int, default=5000)
    ap.add_argument('--timeout', type=int, default=60)
    ap.add_argument('--openphish-url', default=OPENPHISH)
    ap.add_argument('--openphish-mirror', default=OPENPHISH_MIRROR)
    ap.add_argument('--urlhaus-url', default='')
    ap.add_argument('--urlhaus-key', default=os.getenv('URLHAUS_AUTH_KEY', ''))
    args = ap.parse_args()

    urls: list[str] = []
    sources = []
    for name, endpoint in [('OpenPhish', args.openphish_url), ('OpenPhish mirror', args.openphish_mirror)]:
        if not endpoint: continue
        try:
            raw = fetch(endpoint, {'User-Agent': 'PhishGuard-realworld-corpus/1.0', 'Accept': 'text/plain'}, args.timeout)
            extracted = extract_urls(raw)
            urls.extend(extracted)
            sources.append({'name': name, 'url': endpoint, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'entries': len(extracted)})
            if extracted:
                break
        except Exception as exc:
            sources.append({'name': name, 'url': endpoint, 'error': f'{type(exc).__name__}: {exc}'})

    if args.urlhaus_key:
        endpoint = args.urlhaus_url or URLHAUS_TEMPLATE.format(key=args.urlhaus_key)
        try:
            raw = fetch(endpoint, {'User-Agent': 'PhishGuard-realworld-corpus/1.0'}, args.timeout)
            text = raw.decode('utf-8', 'replace')
            reader = csv.DictReader([x for x in text.splitlines() if x and not x.startswith('#')])
            uh = []
            for row in reader:
                u = (row.get('url') or row.get('URL') or '').strip()
                if re.match(r'^https?://', u, re.I): uh.append(u)
            urls.extend(uh)
            sources.append({'name': 'URLhaus', 'url': endpoint, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'entries': len(uh)})
        except Exception as exc:
            sources.append({'name': 'URLhaus', 'url': endpoint, 'error': f'{type(exc).__name__}: {exc}'})

    urls = list(dict.fromkeys(urls))[:max(1, args.limit)]
    if not urls:
        raise SystemExit('No real-world URLs were retrieved; refusing to create a simulated corpus.')

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_role': 'real_world_threat_feed_replay',
        'visited_urls': False,
        'count': len(urls),
        'sources': sources,
        'urls': [{'url': u} for u in urls],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'count': len(urls), 'sources': sources}, indent=2))


if __name__ == '__main__':
    main()
