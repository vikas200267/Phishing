"""Explainable Unicode/IDN diagnostics for URL hostnames.

These are evidence fields only, not verdicts. An internationalized domain can
be legitimate; the model/risk policy must combine these signals with other data.
"""
from __future__ import annotations

import unicodedata
from urllib.parse import urlsplit
from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parents[1]
try:
    _patterns = json.loads((_ROOT / 'data' / 'patterns.json').read_text(encoding='utf-8'))
    _confusables = dict(_patterns.get('confusable_map', {}))
except Exception:
    # This diagnostics helper remains available if the feature-vocabulary file
    # is temporarily absent, but it never substitutes a model or a verdict.
    _confusables = {}

_BIDI_CONTROLS = {chr(x) for x in (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A))}

def _script(ch: str) -> str:
    name = unicodedata.name(ch, '')
    for key in ('LATIN', 'CYRILLIC', 'GREEK', 'HEBREW', 'ARABIC', 'ARMENIAN', 'GEORGIAN', 'HIRAGANA', 'KATAKANA', 'HANGUL', 'CJK'):
        if key in name:
            return key
    if ch.isdigit():
        return 'DIGIT'
    return 'OTHER'


def unicode_host_diagnostics(url: str) -> dict:
    try:
        host = (urlsplit(str(url)).hostname or '').rstrip('.').lower()
    except (ValueError, TypeError):
        host = ''
    punycode_present = 'xn--' in host
    decoded_labels = []
    idna_errors = 0
    for label in host.split('.') if host else []:
        if label.startswith('xn--'):
            try:
                decoded_labels.append(label.encode('ascii').decode('idna'))
            except (UnicodeError, ValueError):
                decoded_labels.append(label)
                idna_errors += 1
        else:
            decoded_labels.append(label)
    unicode_host = '.'.join(decoded_labels)
    try:
        ascii_host = unicode_host.encode('idna').decode('ascii').lower()
        roundtrip_mismatch = int(ascii_host != host)
    except (UnicodeError, ValueError):
        ascii_host = host
        roundtrip_mismatch = 1 if host else 0
        idna_errors += 1
    visible = unicode_host
    scripts = sorted({_script(ch) for ch in visible if ch.isalpha() and _script(ch) not in {'OTHER'}})
    scripts = [x for x in scripts if x != 'DIGIT']
    bidi_count = sum(ch in _BIDI_CONTROLS for ch in visible)
    format_count = sum(unicodedata.category(ch) == 'Cf' for ch in visible)
    confusable_chars = [ch for ch in visible if ch in _confusables]
    return {
        'host': host,
        'unicode_hostname': unicode_host,
        'ascii_hostname': ascii_host,
        'unicode_present': any(ord(ch) > 127 for ch in host),
        'punycode_present': punycode_present,
        'idna_decode_errors': idna_errors,
        'idna_roundtrip_mismatch': roundtrip_mismatch,
        'mixed_script': len(set(scripts) - {'DIGIT', 'OTHER'}) > 1,
        'scripts': scripts,
        'confusable_characters': sorted(set(confusable_chars)),
        'confusable_count': len(confusable_chars),
        'bidi_controls': bidi_count,
        'invisible_format_characters': format_count,
    }
