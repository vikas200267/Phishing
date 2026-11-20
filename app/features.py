from __future__ import annotations
import math, re, unicodedata
from collections import Counter
from difflib import SequenceMatcher
from urllib.parse import urlparse, unquote
import numpy as np

# Every feature is reproducible from the raw URL. No network-only values are used
# in the model path, avoiding training/serving skew.
MODEL_FEATURE_NAMES = [
    'url_length','hostname_length','path_length','query_length','fragment_length',
    'num_digits','digit_ratio','num_letters','letter_ratio','num_dots','num_hyphens',
    'num_underscores','num_slashes','num_at','num_question','num_equal','num_ampersand',
    'num_percent','num_colons','num_semicolons','num_special',
    'entropy','hostname_entropy','path_entropy','query_entropy','subdomain_entropy',
    'subdomain_depth','has_ip','has_punycode','has_port','has_https','has_www',
    'port_value','path_depth','query_param_count','suspicious_token_count',
    'suspicious_host_token_count','unique_token_count','host_token_count','path_token_count',
    'query_token_count','avg_token_length','max_label_length','min_label_length','mean_label_length',
    'max_digit_run','max_hyphen_run','max_alpha_run','has_double_slash_path','has_percent_encoding',
    'has_hex_escape','hex_escape_count','double_encoding','has_userinfo','has_fragment','has_query',
    'shortener','tld_common','tld_risk','brand_token_hits','brand_in_hostname',
    'brand_edit_distance_min','brand_edit_similarity_max','hyphen_ratio_host','digit_ratio_host',
    'special_ratio_host','letter_ratio_host','vowel_ratio_host','label_count',
    'numeric_label_count','mixed_alnum_label_count','non_ascii_count','non_ascii_host_count',
    'unicode_host','mixed_unicode_ascii','repeated_char_run','registered_domain_length',
    'registered_domain_entropy','path_digit_ratio','path_special_ratio','query_digit_ratio',
    'query_special_ratio','root_host_is_common','has_suspicious_token_combo','has_login_payment_combo',
    'has_verify_account_combo','has_security_password_combo','file_extension_suspicious',
    'punycode_label_count','cyrillic_char_count','greek_char_count','cjk_char_count',
    'unicode_script_count','confusable_char_count','mixed_script_host','unicode_ratio_host',
    'tld_length','suspicious_port','brand_in_subdomain_only','brand_in_path_only',
    'registered_domain_is_brand','brand_path_distance','host_path_similarity','host_randomness',
]
FEATURE_NAMES = MODEL_FEATURE_NAMES


# Versioned, data-driven URL vocabularies. These are feature signals, never verdicts.
_PATTERN_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "patterns.json"
try:
    _PATTERNS = __import__("json").loads(_PATTERN_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    raise RuntimeError(f"Missing or invalid data/patterns.json: {exc}") from exc
SUSPICIOUS_TOKENS = frozenset(_PATTERNS.get("suspicious_tokens", []))
SHORTENERS = frozenset(_PATTERNS.get("shorteners", []))
COMMON_TLDS = frozenset(_PATTERNS.get("common_tlds", []))
HIGH_RISK_TLDS = frozenset(_PATTERNS.get("high_risk_tlds", []))
BRANDS = frozenset(_PATTERNS.get("brands", []))
COMMON_ROOT_HOSTS = frozenset(_PATTERNS.get("common_root_hosts", []))
SUSPICIOUS_EXTS = frozenset(_PATTERNS.get("suspicious_extensions", []))
CONFUSABLE_MAP = _PATTERNS.get("confusable_map", {})
TWO_PART_SUFFIXES = frozenset(_PATTERNS.get("two_part_suffixes", []))
SPECIAL_CHARS = set('!$%^&*()_=+[]{};:,?@')

def _script_counts(text: str):
    cyr = greek = cjk = 0
    scripts = set()
    for ch in text:
        name = unicodedata.name(ch, '')
        if 'CYRILLIC' in name:
            cyr += 1; scripts.add('CYRILLIC')
        elif 'GREEK' in name:
            greek += 1; scripts.add('GREEK')
        elif any(x in name for x in ('CJK','HIRAGANA','KATAKANA','HANGUL')):
            cjk += 1; scripts.add('CJK')
        elif ch.isalpha() and ord(ch) < 128:
            scripts.add('LATIN')
    return cyr, greek, cjk, len(scripts)

def _confusable_count(text: str) -> int:
    return sum(ch in CONFUSABLE_MAP for ch in text)

def _normalized_unicode_host(host: str) -> str:
    try:
        return unicodedata.normalize('NFKC', host)
    except Exception:
        return host


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_like_ip(host: str) -> bool:
    if not host:
        return False
    if re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', host):
        return all(0 <= int(x) <= 255 for x in host.split('.'))
    return bool(':' in host and re.fullmatch(r'[0-9a-f:]+', host) is not None)


def _max_run(text: str, pattern: str) -> int:
    return max((len(m.group(0)) for m in re.finditer(pattern, text)), default=0)


def _registrable_domain(labels: list[str]) -> str:
    if not labels:
        return ''
    if len(labels) >= 3 and '.'.join(labels[-2:]) in TWO_PART_SUFFIXES:
        return '.'.join(labels[-3:])
    return '.'.join(labels[-2:]) if len(labels) >= 2 else labels[0]


def normalize_url(raw: str, protocol: str | None = None, domain_name: str | None = None, address: str | None = None) -> str:
    raw = '' if raw is None else str(raw).strip()
    if raw.lower() in {'nan','none'}:
        raw = ''
    if raw.lower().startswith(('http://','https://')):
        return raw
    if domain_name and str(domain_name).strip().lower() not in {'nan','none'}:
        host = str(domain_name).strip()
        path = '' if address is None else str(address).strip()
        if path.lower() in {'nan','none'}:
            path = ''
        scheme = str(protocol or 'http').strip().lower()
        if scheme not in {'http','https'}:
            scheme = 'http'
        suffix = path if not path else (path if path.startswith('/') else '/' + path)
        return f'{scheme}://{host}{suffix}'
    return 'http://' + raw if raw else 'http://'


def _safe_urlparse(url: str):
    try:
        return urlparse(url)
    except (ValueError, TypeError):
        return urlparse("http://invalid.local/")

def extract_url_features(url: str, legacy: dict | None = None) -> dict:
    u = normalize_url(url)
    p = _safe_urlparse(u)
    host = (p.hostname or '').lower().strip('.')
    path = p.path or ''
    query = p.query or ''
    fragment = p.fragment or ''
    decoded = unquote(u).lower()
    host_decoded = unquote(host)
    host_nfkc = _normalized_unicode_host(host_decoded)
    cyr_count, greek_count, cjk_count, unicode_script_count = _script_counts(host_nfkc)
    confusable_count = _confusable_count(host_nfkc)
    mixed_script = int(unicode_script_count >= 2)
    labels = [x for x in host.split('.') if x]
    tld = labels[-1] if labels else ''
    registered_domain = _registrable_domain(labels)
    reg_core = registered_domain.rsplit('.', 1)[0] if '.' in registered_domain else registered_domain
    subdomain_depth = max(0, len(labels) - len(registered_domain.split('.')))
    tokens = re.findall(r'[a-z0-9]+', decoded)
    host_tokens = re.findall(r'[a-z0-9]+', host_decoded)
    q_tokens = re.findall(r'[a-z0-9]+', unquote(query).lower())
    path_tokens = re.findall(r'[a-z0-9]+', unquote(path).lower())
    digit_count = sum(c.isdigit() for c in u)
    letter_count = sum(c.isalpha() for c in u)
    digit_host = sum(c.isdigit() for c in host)
    letter_host = sum(c.isalpha() for c in host)
    special_host = sum((c in SPECIAL_CHARS) or (c in '-_.') for c in host)
    non_ascii = sum(ord(c) > 127 for c in u)
    non_ascii_host = sum(ord(c) > 127 for c in host)
    suspicious_tokens = sum(t in SUSPICIOUS_TOKENS for t in tokens)
    suspicious_host = sum(t in SUSPICIOUS_TOKENS for t in host_tokens)
    max_label = max((len(x) for x in labels), default=0)
    min_label = min((len(x) for x in labels), default=0)
    mean_label = float(np.mean([len(x) for x in labels])) if labels else 0.0
    repeat_run = _max_run(decoded, r'(.)\1{2,}')
    brand_hits = sum(b in tokens for b in BRANDS)
    brand_host = int(any(b in host_tokens for b in BRANDS))
    brand_registered = int(any(reg_core == b for b in BRANDS))
    subdomain_parts = labels[:-len(registered_domain.split('.'))] if subdomain_depth else []
    subdomain_text = '.'.join(subdomain_parts)
    brand_subdomain_only = int(any(b in re.findall(r'[a-z0-9]+', subdomain_text) for b in BRANDS))
    brand_path_only = int((not brand_host) and any(b in path_tokens for b in BRANDS))
    brand_path_sim = max([SequenceMatcher(None, reg_core, t).ratio() for t in path_tokens if t] or [0.0])
    brand_similarities = [SequenceMatcher(None, reg_core, b).ratio() for b in BRANDS if reg_core]
    shortening = int(host in SHORTENERS or any(host.endswith('.' + s) for s in SHORTENERS))
    query_params = 0 if not query else query.count('&') + 1
    try:
        port_value = int(p.port or 0)
    except ValueError:
        port_value = 0
    path_lower = unquote(path).lower()
    last_path_ext = ''
    m_ext = re.search(r'(\.[a-z0-9]{1,6})$', path_lower)
    if m_ext:
        last_path_ext = m_ext.group(1)
    suspicious_ext = int(last_path_ext in SUSPICIOUS_EXTS)
    suspicious_combo = int((brand_host and suspicious_tokens >= 1) or (suspicious_host >= 2))
    login_payment_combo = int(('login' in tokens or 'signin' in tokens) and ('payment' in tokens or 'invoice' in tokens or 'billing' in tokens))
    verify_account_combo = int(('verify' in tokens or 'verification' in tokens or 'confirm' in tokens) and 'account' in tokens)
    security_password_combo = int(('security' in tokens or 'secure' in tokens) and ('password' in tokens or 'credential' in tokens))
    vowel_count_host = sum(c in 'aeiou' for c in host)
    numeric_labels = sum(x.isdigit() for x in labels)
    mixed_labels = sum(int(any(c.isalpha() for c in x) and any(c.isdigit() for c in x)) for x in labels)
    mixed_unicode_ascii = int(non_ascii_host > 0 and any(ord(c) < 128 for c in host))
    f = {
        'url_length': len(u), 'hostname_length': len(host), 'path_length': len(path), 'query_length': len(query), 'fragment_length': len(fragment),
        'num_digits': digit_count, 'digit_ratio': digit_count / max(1, len(u)),
        'num_letters': letter_count, 'letter_ratio': letter_count / max(1, len(u)),
        'num_dots': u.count('.'), 'num_hyphens': u.count('-'), 'num_underscores': u.count('_'),
        'num_slashes': u.count('/'), 'num_at': u.count('@'), 'num_question': u.count('?'), 'num_equal': u.count('='),
        'num_ampersand': u.count('&'), 'num_percent': u.count('%'), 'num_colons': u.count(':'), 'num_semicolons': u.count(';'),
        'num_special': sum(c in SPECIAL_CHARS for c in u), 'entropy': _entropy(decoded), 'hostname_entropy': _entropy(host_decoded),
        'path_entropy': _entropy(unquote(path).lower()), 'query_entropy': _entropy(unquote(query).lower()), 'subdomain_entropy': _entropy('.'.join(labels[:-len(registered_domain.split('.'))]) if subdomain_depth else ''),
        'subdomain_depth': subdomain_depth, 'has_ip': int(_looks_like_ip(host)), 'has_punycode': int('xn--' in host),
        'has_port': int(p.port is not None), 'has_https': int(p.scheme == 'https'), 'has_www': int(host.startswith('www.')), 'port_value': port_value,
        'path_depth': len([x for x in path.split('/') if x]), 'query_param_count': query_params,
        'suspicious_token_count': suspicious_tokens, 'suspicious_host_token_count': suspicious_host,
        'unique_token_count': len(set(tokens)), 'host_token_count': len(host_tokens), 'path_token_count': len(path_tokens), 'query_token_count': len(q_tokens),
        'avg_token_length': float(np.mean([len(x) for x in tokens])) if tokens else 0.0,
        'max_label_length': max_label, 'min_label_length': min_label, 'mean_label_length': mean_label,
        'max_digit_run': _max_run(decoded, r'\d{2,}'), 'max_hyphen_run': _max_run(host, r'-{2,}'), 'max_alpha_run': _max_run(host, r'[A-Za-z]{4,}'),
        'has_double_slash_path': int('//' in path), 'has_percent_encoding': int('%' in u),
        'has_hex_escape': int(bool(re.search(r'%[0-9a-fA-F]{2}', u))), 'hex_escape_count': len(re.findall(r'%[0-9a-fA-F]{2}', u)),
        'double_encoding': int(bool(re.search(r'%25[0-9a-fA-F]{2}', u))), 'has_userinfo': int('@' in (p.netloc or '')),
        'has_fragment': int(bool(fragment)), 'has_query': int(bool(query)), 'shortener': shortening, 'tld_common': int(tld in COMMON_TLDS), 'tld_risk': int(tld in HIGH_RISK_TLDS),
        'brand_token_hits': brand_hits, 'brand_in_hostname': brand_host,
        'brand_edit_distance_min': float(min([(1.0 - s) * max(len(reg_core), 1) for s in brand_similarities] or [1.0])),
        'brand_edit_similarity_max': float(max(brand_similarities, default=0.0)),
        'hyphen_ratio_host': host.count('-') / max(1, len(host)), 'digit_ratio_host': digit_host / max(1, len(host)),
        'special_ratio_host': special_host / max(1, len(host)), 'letter_ratio_host': letter_host / max(1, len(host)),
        'vowel_ratio_host': vowel_count_host / max(1, letter_host), 'label_count': len(labels),
        'numeric_label_count': numeric_labels, 'mixed_alnum_label_count': mixed_labels,
        'non_ascii_count': non_ascii, 'non_ascii_host_count': non_ascii_host, 'unicode_host': int(non_ascii_host > 0), 'mixed_unicode_ascii': mixed_unicode_ascii,
        'repeated_char_run': repeat_run, 'registered_domain_length': len(registered_domain), 'registered_domain_entropy': _entropy(reg_core),
        'path_digit_ratio': sum(c.isdigit() for c in path) / max(1, len(path)), 'path_special_ratio': sum(c in SPECIAL_CHARS for c in path) / max(1, len(path)),
        'query_digit_ratio': sum(c.isdigit() for c in query) / max(1, len(query)), 'query_special_ratio': sum(c in SPECIAL_CHARS for c in query) / max(1, len(query)),
        'root_host_is_common': int(host in COMMON_ROOT_HOSTS), 'has_suspicious_token_combo': suspicious_combo,
        'has_login_payment_combo': login_payment_combo, 'has_verify_account_combo': verify_account_combo,
        'has_security_password_combo': security_password_combo, 'file_extension_suspicious': suspicious_ext,
        'punycode_label_count': sum(1 for x in labels if x.startswith('xn--')),
        'cyrillic_char_count': cyr_count, 'greek_char_count': greek_count, 'cjk_char_count': cjk_count,
        'unicode_script_count': unicode_script_count, 'confusable_char_count': confusable_count,
        'mixed_script_host': mixed_script, 'unicode_ratio_host': non_ascii_host / max(1, len(host)),
        'tld_length': len(tld), 'suspicious_port': int(port_value in {21,22,23,25,110,143,445,3389,5900,8080,8443}),
        'brand_in_subdomain_only': brand_subdomain_only, 'brand_in_path_only': brand_path_only,
        'registered_domain_is_brand': brand_registered,
        'brand_path_distance': float(1.0 - brand_path_sim),
        'host_path_similarity': float(SequenceMatcher(None, reg_core, ' '.join(path_tokens[:8])).ratio()) if path_tokens else 0.0,
        'host_randomness': float(_entropy(reg_core) / max(1.0, math.log2(max(2, len(reg_core))))),
    }
    return f


def urls_to_features(urls):
    import pandas as pd
    rows = [extract_url_features(str(u)) for u in urls]
    return pd.DataFrame.from_records(rows, columns=MODEL_FEATURE_NAMES).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def dataframe_to_features(df):
    import pandas as pd
    if 'url' in df.columns and set(df.columns) <= {'url'}:
        return urls_to_features(df['url'].tolist())
    rows = []
    for r in df.to_dict(orient='records'):
        raw = r.get('url')
        if raw is None or pd.isna(raw) or not str(raw).strip() or str(raw).strip().lower() in {'nan','none'}:
            raw = normalize_url('', r.get('protocol'), r.get('domain_name'), r.get('address'))
        rows.append(extract_url_features(str(raw), r))
    return pd.DataFrame.from_records(rows, columns=MODEL_FEATURE_NAMES).replace([np.inf, -np.inf], 0.0).fillna(0.0)
