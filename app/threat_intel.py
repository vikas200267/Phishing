from __future__ import annotations

import base64
import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx

from .cache import cache_key, get as cache_get, set as cache_set
from .config import settings
from .dns_tls import resolve_host, tls_summary


class ThreatIntel:
    def __init__(self):
        self.client: httpx.AsyncClient | None = None
        self._provider_semaphore = asyncio.Semaphore(32)

    async def start(self):
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.api_timeout_seconds, connect=min(1.5, settings.api_timeout_seconds)),
                limits=httpx.Limits(max_connections=128, max_keepalive_connections=32, keepalive_expiry=30.0),
                headers={'User-Agent': f'PhishGuard/{settings.app_version} security research'},
                follow_redirects=False,
            )

    async def close(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _get(self, *args, **kwargs):
        await self.start()
        assert self.client is not None
        async with self._provider_semaphore:
            return await self.client.get(*args, **kwargs)

    async def _post(self, *args, **kwargs):
        await self.start()
        assert self.client is not None
        async with self._provider_semaphore:
            return await self.client.post(*args, **kwargs)

    async def web_risk(self, url: str) -> dict[str, Any]:
        key = cache_key('webrisk', url)
        cached = cache_get(key) if settings.cache_enabled else None
        if cached is not None:
            return cached
        api_key = settings.web_risk_api_key
        if not api_key:
            return {'enabled': False}
        try:
            params = [
                ('threatTypes', 'MALWARE'),
                ('threatTypes', 'SOCIAL_ENGINEERING'),
                ('threatTypes', 'UNWANTED_SOFTWARE'),
                ('uri', url),
                ('key', api_key),
            ]
            r = await self._get('https://webrisk.googleapis.com/v1/uris:search', params=params)
            r.raise_for_status()
            result = {'enabled': True, 'match': r.json().get('threat', {})}
            if settings.cache_enabled:
                cache_set(key, result, ttl=max(60, settings.cache_confident_ttl_seconds))
            return result
        except Exception as e:
            return {'enabled': True, 'error': type(e).__name__}

    async def safe_browsing(self, urls: list[str]) -> dict[str, Any]:
        if not settings.safe_browsing_api_key:
            return {'enabled': False}
        cache_value = '\n'.join(sorted(set(urls[:500])))
        key = cache_key('safe-browsing', cache_value)
        cached = cache_get(key) if settings.cache_enabled else None
        if cached is not None:
            return cached
        body = {
            'client': {'clientId': 'phishguard', 'clientVersion': settings.app_version},
            'threatInfo': {
                'threatTypes': ['MALWARE', 'SOCIAL_ENGINEERING', 'UNWANTED_SOFTWARE', 'POTENTIALLY_HARMFUL_APPLICATION'],
                'platformTypes': ['ANY_PLATFORM'],
                'threatEntryTypes': ['URL'],
                'threatEntries': [{'url': u} for u in urls[:500]],
            },
        }
        try:
            r = await self._post(
                'https://safebrowsing.googleapis.com/v4/threatMatches:find',
                params={'key': settings.safe_browsing_api_key},
                json=body,
            )
            r.raise_for_status()
            result = {'enabled': True, 'matches': r.json().get('matches', [])}
            if settings.cache_enabled:
                cache_set(key, result, ttl=max(60, settings.cache_confident_ttl_seconds))
            return result
        except Exception as e:
            return {'enabled': True, 'error': type(e).__name__}

    async def virustotal(self, url: str) -> dict[str, Any]:
        if not settings.vt_api_key:
            return {'enabled': False}
        key = cache_key('virustotal', url)
        cached = cache_get(key) if settings.cache_enabled else None
        if cached is not None:
            return cached
        url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip('=')
        try:
            r = await self._get(
                f'https://www.virustotal.com/api/v3/urls/{url_id}',
                headers={'x-apikey': settings.vt_api_key, 'accept': 'application/json'},
            )
            r.raise_for_status()
            attrs = r.json().get('data', {}).get('attributes', {})
            stats = attrs.get('last_analysis_stats', {})
            result = {
                'enabled': True,
                'malicious': int(stats.get('malicious', 0) or 0),
                'suspicious': int(stats.get('suspicious', 0) or 0),
                'harmless': int(stats.get('harmless', 0) or 0),
                'reputation': attrs.get('reputation'),
            }
            if settings.cache_enabled:
                cache_set(key, result, ttl=max(60, settings.cache_confident_ttl_seconds))
            return result
        except Exception as e:
            return {'enabled': True, 'error': type(e).__name__}

    async def urlscan_search(self, domain: str) -> dict[str, Any]:
        if not settings.urlscan_api_key or not domain:
            return {'enabled': False}
        key = cache_key('urlscan', domain.lower())
        cached = cache_get(key) if settings.cache_enabled else None
        if cached is not None:
            return cached
        try:
            r = await self._get(
                'https://urlscan.io/api/v1/search',
                params={'q': f'domain:{domain}', 'size': 10},
                headers={'api-key': settings.urlscan_api_key},
            )
            r.raise_for_status()
            data = r.json()
            result = {'enabled': True, 'total': data.get('total', 0), 'results': data.get('results', [])[:10]}
            if settings.cache_enabled:
                cache_set(key, result, ttl=max(60, settings.cache_confident_ttl_seconds))
            return result
        except Exception as e:
            return {'enabled': True, 'error': type(e).__name__}

    async def phishtank(self, url: str) -> dict[str, Any]:
        if not settings.phishtank_app_key:
            return {'enabled': False}
        key = cache_key('phishtank', url)
        cached = cache_get(key) if settings.cache_enabled else None
        if cached is not None:
            return cached
        try:
            r = await self._post(
                'https://checkurl.phishtank.com/checkurl/',
                data={'url': url, 'format': 'json', 'app_key': settings.phishtank_app_key},
            )
            r.raise_for_status()
            data = r.json().get('results', {})
            result = {
                'enabled': True,
                'in_database': data.get('in_database'),
                'verified': data.get('verified'),
                'valid': data.get('valid'),
                'phish_id': data.get('phish_id'),
            }
            if settings.cache_enabled:
                cache_set(key, result, ttl=max(60, settings.cache_confident_ttl_seconds))
            return result
        except Exception as e:
            return {'enabled': True, 'error': type(e).__name__}

    async def enrich_url(self, url: str, include_remote: bool = True) -> dict[str, Any]:
        p = urlparse(url)
        host = p.hostname or ''
        # Cache by origin (not path), because DNS/TLS properties are origin-level.
        # The TTL remains short so certificate/DNS changes don't become permanent.
        origin = f'{p.scheme.lower()}://{host.lower()}:{p.port or (443 if p.scheme.lower() == "https" else 80)}'
        local_key = cache_key('dns-tls-origin', origin)
        local_cached = cache_get(local_key) if settings.cache_enabled else None
        if isinstance(local_cached, dict) and 'dns' in local_cached and 'tls' in local_cached:
            # Copy before appending provider results so remote fields never mutate
            # the cached origin-only DNS/TLS record.
            result = dict(local_cached)
        else:
            # DNS and TLS helpers perform blocking socket operations. Keep them off
            # the event loop so one slow resolver/handshake cannot stall unrelated API
            # requests. SSRF/private-address checks remain inside resolve_host/tls_summary.
            dns = await asyncio.to_thread(resolve_host, host)
            if p.scheme == 'https':
                tls = await asyncio.to_thread(tls_summary, url)
            else:
                tls = {'ok': False, 'reason': 'not_https'}
            result = {'dns': dns, 'tls': tls}
            local_ttl = max(30, settings.cache_uncertain_ttl_seconds)
            if dns.get('ok') and (tls.get('ok') or tls.get('reason') == 'not_https'):
                local_ttl = max(local_ttl, min(300, settings.cache_confident_ttl_seconds))
            if settings.cache_enabled:
                cache_set(local_key, result, ttl=local_ttl)
        dns = result.get('dns', {}) or {}
        if not include_remote:
            return result
        # Never hand private/local destinations to third-party scanners. This
        # prevents the enrichment feature from becoming a remote pivot/SSRF aid.
        if not dns.get('ok') and dns.get('reason') in {'non_public_host','private_or_local_resolution'}:
            result['remote_enrichment'] = {'enabled': False, 'blocked': True, 'reason': dns.get('reason')}
            return result
        web_risk, safe, vt, us, pt = await asyncio.gather(
            self.web_risk(url), self.safe_browsing([url]), self.virustotal(url), self.urlscan_search(host), self.phishtank(url)
        )
        result.update({'web_risk': web_risk, 'safe_browsing': safe, 'virustotal': vt, 'urlscan': us, 'phishtank': pt})
        return result
