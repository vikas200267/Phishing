from __future__ import annotations
import csv
import json
import os
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit, urlunsplit
import ipaddress
from .features import normalize_url
from .config import settings

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR=ROOT/'data'/'feeds'/'live'
SOURCE_DIR=Path(settings.feed_dir) if settings.feed_dir else DEFAULT_SOURCE_DIR
PHISHING_FEED_PATH=SOURCE_DIR/'openphish.csv'
URLHAUS_FEED_PATH=SOURCE_DIR/'urlhaus.csv'
_LOCK=RLock(); LOCAL_PHISHING_URLS=frozenset(); FEED_METADATA={}; _MTIMES={}; _NEXT_EXPIRY=None


def _utcnow():
    return datetime.now(timezone.utc)


def canonical_url(url: str)->str:
    """Normalize only URL-equivalent pieces while preserving path/query case.

    Fragments are omitted because they are not sent to web servers. Default
    ports and hostname casing are normalized. Query ordering, path case, and
    percent escapes are preserved to avoid turning a path-specific threat into
    a broader domain verdict.
    """
    u=normalize_url(url)
    try:
        p=urlsplit(u)
        scheme=p.scheme.lower()
        raw_host=(p.hostname or '').rstrip('.').lower()
        if not raw_host:
            return u
        try:
            ipaddress.ip_address(raw_host)
            host=raw_host
        except ValueError:
            try:
                host=raw_host.encode('idna').decode('ascii').lower()
            except (UnicodeError, ValueError):
                host=raw_host
        if ':' in host and not host.startswith('['):
            host=f'[{host}]'
        try:
            port=p.port
        except ValueError:
            return u
        if (scheme=='http' and port==80) or (scheme=='https' and port==443):
            port=None
        netloc=host + (f':{port}' if port is not None else '')
        path=p.path or '/'
        return urlunsplit((scheme,netloc,path,p.query,''))
    except (ValueError,TypeError): return u


def _load_set(path: Path,label:int)->set[str]:
    out=set()
    if not path.exists(): return out
    try:
        with path.open('r',newline='',encoding='utf-8',errors='replace') as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or 'url' not in reader.fieldnames: return out
            for row in reader:
                if 'label' in row:
                    try:
                        if int(float(row.get('label') or -1)) != label: continue
                    except (ValueError,TypeError): continue
                u=canonical_url(row.get('url') or '')
                if len(u)>=8: out.add(hashlib.sha256(u.encode('utf-8')).digest())
    except (OSError,ValueError):
        return set()
    return out


def _mtime(path: Path):
    try: return path.stat().st_mtime_ns
    except OSError: return None


def _feed_observation_meta(path: Path):
    latest=None; rows=0
    try:
        with path.open('r',newline='',encoding='utf-8',errors='replace') as f:
            for row in csv.DictReader(f):
                rows += 1
                ts=row.get('collected_at') or row.get('observed_at')
                if ts:
                    try:
                        dt=datetime.fromisoformat(ts.replace('Z','+00:00')).astimezone(timezone.utc)
                        if latest is None or dt>latest: latest=dt
                    except ValueError:
                        pass
    except (OSError,ValueError):
        return {'rows':rows,'latest':None}
    return {'rows':rows,'latest':latest}


def reload_local_intel()->dict:
    global LOCAL_PHISHING_URLS,FEED_METADATA,_MTIMES,_NEXT_EXPIRY
    feeds=[PHISHING_FEED_PATH,URLHAUS_FEED_PATH]
    with _LOCK:
        ph=set(); source_meta=[]; now=_utcnow(); newest=None; expiries=[]
        for path in feeds:
            if not path.exists():
                continue
            info=_feed_observation_meta(path)
            signed_age=None if info['latest'] is None else (now-info['latest']).total_seconds()
            age=None if signed_age is None else max(0.0, signed_age)
            fresh=bool(signed_age is not None and -300 <= signed_age < settings.local_intel_max_age_seconds)
            source_meta.append({'name':path.stem,'rows':info['rows'],'collected_at_max':info['latest'].isoformat() if info['latest'] else None,'age_seconds':round(age,3) if age is not None else None,'fresh':fresh})
            if fresh:
                ph.update(_load_set(path,1))
                expiries.append(info['latest'] + timedelta(seconds=settings.local_intel_max_age_seconds))
            if info['latest'] and (newest is None or info['latest']>newest): newest=info['latest']
        LOCAL_PHISHING_URLS=frozenset(ph)
        _MTIMES={str(p):_mtime(p) for p in feeds}
        _NEXT_EXPIRY=min(expiries) if expiries else None
        meta={'paths':[str(p) for p in feeds if p.exists()], 'entries':len(LOCAL_PHISHING_URLS), 'sources':source_meta, 'max_age_seconds':settings.local_intel_max_age_seconds}
        if newest:
            meta['collected_at_max']=newest.isoformat(); meta['age_seconds']=round(max(0.0,(now-newest).total_seconds()),3)
        else:
            meta['age_seconds']=None
        meta['stale']=not any(x['fresh'] for x in source_meta)
        FEED_METADATA=meta
        return dict(meta)


def maybe_reload_local_intel()->bool:
    with _LOCK: old=dict(_MTIMES); expiry=_NEXT_EXPIRY
    current={str(p):_mtime(p) for p in [PHISHING_FEED_PATH,URLHAUS_FEED_PATH]}
    if current!=old or (expiry is not None and _utcnow() >= expiry):
        reload_local_intel(); return True
    return False

reload_local_intel()


def local_phishing(url:str)->bool:
    with _LOCK:
        expiry=_NEXT_EXPIRY
    if expiry is not None and _utcnow() >= expiry:
        reload_local_intel()
    canonical = canonical_url(url)
    return hashlib.sha256(canonical.encode('utf-8')).digest() in LOCAL_PHISHING_URLS


def local_intel_metadata()->dict:
    maybe_reload_local_intel()
    with _LOCK: result = dict(FEED_METADATA)
    result['sources'] = [dict(source) for source in result.get('sources', [])]
    now = _utcnow()
    for source in result['sources']:
        if source.get('collected_at_max'):
            collected=datetime.fromisoformat(source['collected_at_max'])
            source['age_seconds']=round(max(0.0, (now-collected).total_seconds()),3)
    if result.get('collected_at_max'):
        collected=datetime.fromisoformat(result['collected_at_max'])
        result['age_seconds']=round(max(0.0, (now-collected).total_seconds()),3)
    status_path = SOURCE_DIR / 'refresh_status.json'
    try:
        status = json.loads(status_path.read_text(encoding='utf-8'))
        providers = status.get('providers', {}) if isinstance(status, dict) else {}
        # Expose operational state without returning provider error text or any
        # credential-bearing URL that may have been persisted by old versions.
        result['refresh'] = {
            'checked_at': status.get('checked_at'),
            'providers': {
                name: {
                    'ok': bool(value.get('ok')),
                    'skipped': bool(value.get('skipped', False)),
                    'reason': value.get('reason') if value.get('skipped') else None,
                }
                for name, value in providers.items()
                if isinstance(value, dict)
            },
        }
    except (OSError, ValueError, TypeError):
        result['refresh'] = {'checked_at': None, 'providers': {}, 'status': 'unavailable'}
    return result
