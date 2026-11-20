from __future__ import annotations
import re
from email import policy
from email.parser import BytesParser
from html import unescape
from urllib.parse import urlparse

URL_RE = re.compile(r"https?://[^\s<>'\"\]\)]+", re.I)
URGENCY = {"urgent","immediately","action required","verify","suspended","expire","final notice","security alert","password"}
FINANCIAL = {"bank","payment","invoice","card","wallet","account","transfer","refund","tax"}

def _text(v) -> str:
    return "" if v is None else str(v)

def extract_email_features(*, subject: str = "", sender: str = "", reply_to: str = "", return_path: str = "", body: str = "", headers: dict | None = None) -> dict:
    headers = headers or {}
    body_plain = unescape(re.sub(r"<[^>]+>", " ", _text(body)))
    lower = " ".join([_text(subject), body_plain]).lower()
    urls = URL_RE.findall(_text(body) + " " + _text(subject))
    sender_domain = _text(sender).split("@")[-1].lower().strip(">< ") if "@" in _text(sender) else ""
    reply_domain = _text(reply_to).split("@")[-1].lower().strip(">< ") if "@" in _text(reply_to) else ""
    return_domain = _text(return_path).split("@")[-1].lower().strip(">< ") if "@" in _text(return_path) else ""
    auth = {k.lower(): _text(v).lower() for k,v in headers.items()}
    spf = auth.get("authentication-results", "")
    return {
        "subject_len": len(_text(subject)),
        "body_len": len(body_plain),
        "url_count": len(urls),
        "unique_domain_count": len({urlparse(u).hostname for u in urls if urlparse(u).hostname}),
        "html_ratio": (_text(body).count("<") / max(1, len(_text(body)))),
        "digit_ratio": sum(c.isdigit() for c in lower) / max(1, len(lower)),
        "exclamation_count": lower.count("!"),
        "urgency_hits": sum(term in lower for term in URGENCY),
        "financial_hits": sum(term in lower for term in FINANCIAL),
        "sender_reply_mismatch": int(bool(sender_domain and reply_domain and sender_domain != reply_domain)),
        "sender_return_mismatch": int(bool(sender_domain and return_domain and sender_domain != return_domain)),
        "spf_pass": int("spf=pass" in spf),
        "dkim_pass": int("dkim=pass" in spf),
        "dmarc_pass": int("dmarc=pass" in spf),
        "spf_fail": int("spf=fail" in spf),
        "dkim_fail": int("dkim=fail" in spf),
        "dmarc_fail": int("dmarc=fail" in spf),
        "has_attachment": int(bool(headers.get("content-disposition"))),
    }

def parse_eml(raw_bytes: bytes) -> dict:
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = _text(msg.get("subject"))
    sender = _text(msg.get("from"))
    reply_to = _text(msg.get("reply-to"))
    return_path = _text(msg.get("return-path"))
    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() in {"text/plain", "text/html"}:
                try: body_parts.append(_text(part.get_content()))
                except Exception: pass
    else:
        try: body_parts.append(_text(msg.get_content()))
        except Exception: pass
    return {"subject": subject, "sender": sender, "reply_to": reply_to, "return_path": return_path, "body": "\n".join(body_parts), "headers": dict(msg.items())}
