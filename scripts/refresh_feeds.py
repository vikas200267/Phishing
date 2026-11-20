from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import tempfile
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'data' / 'feeds' / 'live'


def _atomic_write(path: pathlib.Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _record_refresh_status(outdir: pathlib.Path, source: str, result: dict) -> None:
    path = outdir / 'refresh_status.json'
    try:
        previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    except (OSError, ValueError):
        previous = {}
    providers = previous.get('providers', {})
    if not isinstance(providers, dict):
        providers = {}
    providers[source] = result
    _atomic_json(path, {
        'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'providers': providers,
    })


def _archive_snapshot(outdir: pathlib.Path, source: str, rows: list[dict[str, str]], fields: list[str], captured_at: str) -> pathlib.Path:
    snapshots = outdir / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    moment = dt.datetime.fromisoformat(captured_at.replace('Z', '+00:00')).astimezone(dt.timezone.utc)
    target = snapshots / f'{source}_{moment.strftime("%Y%m%dT%H%M%S%fZ")}.csv'
    _atomic_write(target, rows, fields)
    _atomic_json(target.with_suffix(target.suffix + '.meta.json'), {
        'source': source,
        'captured_at': captured_at,
        'entries': len(rows),
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'listed_urls_visited': False,
    })
    retention_days = max(1, int(os.getenv('FEED_SNAPSHOT_RETENTION_DAYS', '90')))
    cutoff = moment.timestamp() - retention_days * 86400
    for old in snapshots.glob(f'{source}_*.csv'):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
                old.with_suffix(old.suffix + '.meta.json').unlink(missing_ok=True)
        except OSError:
            continue
    return target


def _safe_source(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.scheme and parsed.hostname:
            return f'{parsed.scheme.lower()}://{parsed.hostname.lower()}'
    except ValueError:
        pass
    return 'configured feed endpoint'


def _fetch(url: str, headers: dict[str, str], timeout: int = 60):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers), r.geturl()


def refresh_openphish(out: pathlib.Path, url: str, mirror: str | None):
    urls = [x for x in [url, mirror] if x]
    errors = []
    for endpoint in urls:
        try:
            raw, headers, resolved = _fetch(endpoint, {'User-Agent': 'PhishGuard feed refresher', 'Accept': 'text/plain'})
            seen = set(); rows = []
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            for line in raw.decode('utf-8', 'replace').splitlines():
                u = line.strip()
                if u and u.lower().startswith(('http://', 'https://')) and u not in seen:
                    seen.add(u); rows.append({'url': u, 'label': '1', 'source': 'openphish', 'collected_at': now})
            if not rows:
                raise RuntimeError('feed response contained no valid HTTP(S) URLs; preserving prior feed')
            fields = ['url', 'label', 'source', 'collected_at']
            _atomic_write(out, rows, fields)
            snapshot = _archive_snapshot(out.parent, 'openphish', rows, fields, now)
            _atomic_json(out.with_suffix(out.suffix + '.meta.json'), {
                'source': 'openphish', 'source_url': _safe_source(resolved), 'retrieved_at': now,
                'sha256': hashlib.sha256(raw).hexdigest(), 'entries': len(rows),
                'listed_urls_visited': False, 'snapshot_path': str(snapshot.relative_to(out.parent)),
                'etag': headers.get('ETag'), 'last_modified': headers.get('Last-Modified')
            })
            return {'ok': True, 'entries': len(rows), 'source_url': _safe_source(resolved), 'snapshot': str(snapshot.relative_to(out.parent))}
        except Exception as e:
            errors.append(type(e).__name__)
    return {'ok': False, 'errors': errors}


def refresh_urlhaus(out: pathlib.Path, url_template: str, auth_key: str | None):
    if not auth_key:
        return {'ok': False, 'skipped': True, 'reason': 'URLHAUS_AUTH_KEY not configured'}
    endpoint = url_template.format(key=auth_key)
    try:
        raw, _, resolved = _fetch(endpoint, {'User-Agent': 'PhishGuard feed refresher'}, timeout=90)
        text = raw.decode('utf-8', 'replace'); lines = text.splitlines()
        start = next((i for i, line in enumerate(lines) if line and not line.startswith('#')), 0)
        reader = csv.DictReader(lines[start:]); fields = reader.fieldnames or []
        url_key = next((x for x in ('url', 'URL') if x in fields), None)
        if not url_key: raise RuntimeError('URLhaus export did not contain a URL column')
        now = dt.datetime.now(dt.timezone.utc).isoformat(); rows = []
        for row in reader:
            u = (row.get(url_key) or '').strip()
            if u.startswith(('http://', 'https://')):
                rows.append({'url': u, 'label': '1', 'source': 'urlhaus', 'collected_at': now})
        if not rows:
            raise RuntimeError('URLhaus response contained no valid HTTP(S) URLs; preserving prior feed')
        fields = ['url', 'label', 'source', 'collected_at']
        _atomic_write(out, rows, fields)
        snapshot = _archive_snapshot(out.parent, 'urlhaus', rows, fields, now)
        safe_resolved = _safe_source(resolved)
        _atomic_json(out.with_suffix(out.suffix + '.meta.json'), {
            'source': 'urlhaus', 'source_url': safe_resolved, 'retrieved_at': now,
            'sha256': hashlib.sha256(raw).hexdigest(), 'entries': len(rows), 'listed_urls_visited': False,
            'snapshot_path': str(snapshot.relative_to(out.parent))
        })
        return {'ok': True, 'entries': len(rows), 'source_url': safe_resolved, 'snapshot': str(snapshot.relative_to(out.parent))}
    except Exception as e:
        return {'ok': False, 'error': type(e).__name__}


def main():
    ap = argparse.ArgumentParser(description='Refresh PhishGuard threat feeds without visiting listed URLs')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--feed-dir', default=os.getenv('PHISHGUARD_FEED_DIR', str(DEFAULT_DIR)))
    ap.add_argument('--openphish-url', default=os.getenv('OPENPHISH_FEED_URL', 'https://www.openphish.com/feed.txt'))
    ap.add_argument('--openphish-mirror', default=os.getenv('OPENPHISH_MIRROR_URL', 'https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt'))
    ap.add_argument('--openphish-interval', type=int, default=int(os.getenv('OPENPHISH_REFRESH_SECONDS', '43200')))
    ap.add_argument('--urlhaus-template', default=os.getenv('URLHAUS_EXPORT_URL_TEMPLATE', 'https://urlhaus-api.abuse.ch/v2/files/exports/{key}/recent.csv'))
    ap.add_argument('--urlhaus-key', default=os.getenv('URLHAUS_AUTH_KEY', ''))
    ap.add_argument('--urlhaus-interval', type=int, default=int(os.getenv('URLHAUS_REFRESH_SECONDS', '300')))
    ap.add_argument('--retry-seconds', type=int, default=int(os.getenv('FEED_RETRY_SECONDS', '900')))
    a = ap.parse_args(); outdir = pathlib.Path(a.feed_dir)
    next_open = next_urlh = 0.0
    while True:
        now = time.time()
        if now >= next_open:
            result = {'openphish': refresh_openphish(outdir/'openphish.csv', a.openphish_url, a.openphish_mirror)}
            _record_refresh_status(outdir, 'openphish', result['openphish'])
            print(json.dumps(result, sort_keys=True), flush=True)
            interval = max(300, a.openphish_interval)
            retry = max(300, a.retry_seconds)
            next_open = now + (interval if result['openphish'].get('ok') else min(interval, retry))
        if now >= next_urlh:
            result = {'urlhaus': refresh_urlhaus(outdir/'urlhaus.csv', a.urlhaus_template, a.urlhaus_key)}
            _record_refresh_status(outdir, 'urlhaus', result['urlhaus'])
            print(json.dumps(result, sort_keys=True), flush=True)
            interval = max(300, a.urlhaus_interval)
            retry = max(300, a.retry_seconds)
            if result['urlhaus'].get('skipped'):
                next_urlh = now + interval
            else:
                next_urlh = now + (interval if result['urlhaus'].get('ok') else min(interval, retry))
        if a.once: break
        time.sleep(5)


if __name__ == '__main__':
    main()
