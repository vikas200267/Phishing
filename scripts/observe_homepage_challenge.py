from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SameSiteRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        old_host = (old.hostname or '').lower().rstrip('.')
        new_host = (new.hostname or '').lower().rstrip('.')
        if new.scheme.lower() != 'https' or not old_host or not new_host:
            return None
        if _registrable(old_host) != _registrable(new_host):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _registrable(host: str) -> str:
    # The public-suffix aware project helper handles multi-label suffixes.
    from scripts.audit_dataset_leakage import registrable
    return registrable('https://' + host + '/')


def observe(url: str, timeout: float) -> dict:
    observed_at = datetime.now(timezone.utc).isoformat()
    opener = urllib.request.build_opener(SameSiteRedirect())
    req = urllib.request.Request(url, headers={
        'User-Agent': 'PhishGuard homepage verification/1.0',
        'Accept': 'text/html,application/xhtml+xml',
        'Range': 'bytes=0-4095',
    })
    try:
        with opener.open(req, timeout=timeout) as response:
            final_url = response.geturl()
            status = int(response.status)
            content_type = response.headers.get('Content-Type', '')
            # Consume only a tiny homepage prefix; this tool never follows links.
            response.read(4096)
        same_site = _registrable(urlsplit(url).hostname or '') == _registrable(urlsplit(final_url).hostname or '')
        verified = status in range(200, 300) and same_site and 'text/html' in content_type.lower()
        return {'url': url, 'observed_at': observed_at, 'status': status,
                'final_url': final_url, 'content_type': content_type, 'verified': verified,
                'error_type': None}
    except urllib.error.HTTPError as exc:
        # An HTTP response, including a rate-limit or access-denied response,
        # still records a real timestamped observation of the exact origin.
        final_url = exc.geturl()
        status = int(exc.code)
        try:
            same_site = _registrable(urlsplit(url).hostname or '') == _registrable(urlsplit(final_url).hostname or '')
        except Exception:
            same_site = False
        return {'url': url, 'observed_at': observed_at,
                'status': status, 'final_url': final_url,
                'content_type': exc.headers.get('Content-Type', '') if exc.headers else '',
                'verified': False, 'error_type': type(exc).__name__}
    except Exception as exc:
        return {'url': url, 'observed_at': observed_at, 'status': None, 'final_url': None,
                'content_type': None, 'verified': False,
                'error_type': type(exc).__name__, 'error_detail': str(exc)[:240]}


def main() -> int:
    parser = argparse.ArgumentParser(description='Fetch only the curated official homepage challenge roots and record real observation times.')
    parser.add_argument('--csv', type=Path, default=ROOT/'data/realworld/legitimate_challenge.csv')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.timeout <= 0:
        raise ValueError('workers must be 1..16 and timeout positive')

    with args.csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0]) if rows else []
    if not rows or not {'url', 'label', 'source', 'observed_at'}.issubset(fields):
        raise ValueError('Challenge CSV must contain url,label,source,observed_at')
    allowed = set()
    with (ROOT/'data/realworld/trusted_homepages.csv').open(newline='', encoding='utf-8') as f:
        allowed = {r['hostname'].strip().lower() for r in csv.DictReader(f)}
    urls = [r['url'] for r in rows]
    if len(urls) != len(set(urls)):
        raise ValueError('Challenge URLs must be unique')
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.hostname not in allowed:
            raise ValueError(f'Refusing to fetch URL outside the curated HTTPS homepage list: {url}')

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(observe, url, args.timeout) for url in urls]
        for future in as_completed(futures):
            result = future.result()
            results[result['url']] = result

    for row in rows:
        evidence = results[row['url']]
        row['observed_at'] = evidence['observed_at']
        row['verification_status'] = 'verified_http_html' if evidence['verified'] else ('http_response_non_html_or_error' if evidence['status'] is not None else 'observation_attempt_failed')
    output_fields = fields + [c for c in ('verification_status',) if c not in fields]
    with args.csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.csv.with_suffix(args.csv.suffix+'.meta.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    coverage = sum(bool(r['observed_at']) for r in rows) / len(rows)
    metadata.update({
        'sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        'rows': len(rows),
        'verification_date': datetime.now(timezone.utc).date().isoformat(),
        'verification_method': 'Direct HTTPS GET of homepage root only; same-registrable-domain HTTPS redirects only; first 4096 bytes read; no links followed. observed_at records the authentic attempt time; verification_status distinguishes verified page responses from failed TLS/network attempts.',
        'observation_time_known': coverage >= 0.8,
        'observed_at_coverage': coverage,
        'successful_html_observations': sum(r['verified'] for r in results.values()),
        'listed_urls_visited': True,
        'observations': results,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'verified': sum(r['verified'] for r in results.values()),
                      'observation_time_coverage': coverage, 'results': list(results.values())}, indent=2))
    return 0 if coverage >= 0.8 else 2


if __name__ == '__main__':
    raise SystemExit(main())
