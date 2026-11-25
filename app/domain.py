"""Deterministic registrable-domain groups for leakage checks."""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import tldextract

# Use the snapshot shipped with the pinned package. Never fetch a changing
# suffix list during training or evaluation; hosted tenant suffixes count too.
_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
DOMAIN_GROUPING_VERSION = 'tldextract-5.4.0-bundled-psl-private'


def registrable_domain(url: str) -> str:
    try:
        host = (urlsplit(str(url)).hostname or '').rstrip('.').lower()
        if not host:
            return ''
        try:
            return str(ipaddress.ip_address(host))
        except ValueError:
            host = host.encode('idna').decode('ascii')
        parsed = _extract(host)
        domain = parsed.top_domain_under_public_suffix
        if domain:
            return domain
        # For unknown suffixes, group sibling subdomains conservatively.
        labels = [part for part in host.split('.') if part]
        return '.'.join(labels[-2:])
    except (ValueError, TypeError, UnicodeError):
        return ''
