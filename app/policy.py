from __future__ import annotations
import ipaddress
import re
from urllib.parse import urlparse


def valid_http_url(url: str) -> bool:
    """Accept only syntactically valid public-input HTTP(S) URLs.

    This function does not classify a URL as safe or unsafe. It is input
    validation only, so there is no hard-coded benign verdict path.
    """
    try:
        raw = str(url).strip()
        if not raw or any(ord(c) < 0x20 or c.isspace() for c in raw):
            return False
        p = urlparse(raw)
        if p.scheme.lower() not in {'http', 'https'} or not p.hostname:
            return False
        if p.username is not None or p.password is not None:
            return False
        port = p.port
        if port is not None and not (1 <= port <= 65535):
            return False
        host = p.hostname.rstrip('.').lower()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            ascii_host = host.encode('idna').decode('ascii')
            labels = ascii_host.split('.')
            if not host or len(ascii_host) > 253 or any(
                not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label)
                for label in labels
            ):
                return False
        return True
    except (ValueError, TypeError, UnicodeError):
        return False
