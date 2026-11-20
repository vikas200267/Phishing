"""Narrow exceptions for independently identified official homepages.

These exceptions apply only to the HTTPS origin root. They do not trust arbitrary
subdomains, deep links, query strings, or a model's exact threat-intelligence hit.
The model score is retained separately by the API for auditability.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
TRUSTED_HOMEPAGES = ROOT / 'data/realworld/trusted_homepages.csv'


@lru_cache(maxsize=1)
def trusted_hosts() -> frozenset[str]:
    if not TRUSTED_HOMEPAGES.exists():
        return frozenset()
    with TRUSTED_HOMEPAGES.open(newline='', encoding='utf-8') as f:
        return frozenset(
            row['hostname'].strip().lower().rstrip('.')
            for row in csv.DictReader(f)
            if row.get('hostname', '').strip()
        )


def is_trusted_official_homepage(url: str) -> bool:
    """Match only exact listed HTTPS hosts at the unparameterized origin root."""
    try:
        parsed = urlsplit(str(url).strip())
        host = (parsed.hostname or '').lower().rstrip('.')
        return bool(
            parsed.scheme.lower() == 'https'
            and host in trusted_hosts()
            and parsed.username is None
            and parsed.password is None
            and parsed.port in (None, 443)
            and parsed.path in ('', '/')
            and not parsed.query
            and not parsed.fragment
        )
    except (ValueError, TypeError):
        return False
