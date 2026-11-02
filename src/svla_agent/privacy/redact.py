"""Redact PII and on-chip secrets from log content *before* it is indexed,
sent to an LLM, or written into a report.

The rules are deliberately conservative: register dumps and ordinary hex values
are left alone, while e-mail addresses, IP/MAC addresses, home-directory user
names, device serial/fuse identifiers and anything that looks like a credential
are replaced with a typed placeholder such as ``<EMAIL>`` or ``<SECRET>``.
"""

from __future__ import annotations

import re

from ..models import Event

_SECRET_KEYS = re.compile(
    r"^(api[_-]?key|token|secret|password|passwd|pwd|bearer|auth|credential|"
    r"otp|private[_-]?key|fuse_key|hw_key|efuse_key)$",
    re.IGNORECASE,
)
_IDENTITY_KEYS = re.compile(
    r"^(sn|serial|serial_no|chip_id|die_id|fuse_id|uid|device_id|imei|mac)$",
    re.IGNORECASE,
)

# (placeholder, pattern) applied in order to free text.
_TEXT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("<EMAIL>", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("<IPV4>", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("<MAC>", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")),
    ("<HOME>", re.compile(r"(?:/Users|/home|C:\\\\Users)/[^/\s\\]+")),
    (
        "<SECRET>",
        re.compile(
            r"(?i)\b(?:api[_-]?key|token|secret|password|passwd|pwd|bearer|private[_-]?key)"
            r"\s*[=:]\s*\S+"
        ),
    ),
    (
        "<ID>",
        re.compile(r"(?i)\b(?:sn|serial|chip_id|die_id|fuse_id|uid|device_id|imei)\s*[=:]\s*\S+"),
    ),
]


def _keep_key(match: re.Match[str], placeholder: str) -> str:
    """For ``key=value`` rules keep the key so the line stays readable."""
    text = match.group(0)
    for sep in ("=", ":"):
        if sep in text:
            key = text.split(sep, 1)[0].strip()
            return f"{key}{sep}{placeholder}"
    return placeholder


def redact_text(text: str) -> str:
    """Return ``text`` with PII / secrets replaced by placeholders."""
    if not text:
        return text
    out = text
    for placeholder, pattern in _TEXT_RULES:
        if placeholder in {"<SECRET>", "<ID>"}:

            def _sub(match: re.Match[str], _p: str = placeholder) -> str:
                return _keep_key(match, _p)

            out = pattern.sub(_sub, out)
        else:
            out = pattern.sub(placeholder, out)
    return out


def redact_kv(kv: dict[str, str]) -> dict[str, str]:
    """Redact sensitive values in a parsed key/value dict (keys are kept)."""
    clean: dict[str, str] = {}
    for key, value in kv.items():
        if _SECRET_KEYS.match(key):
            clean[key] = "<SECRET>"
        elif _IDENTITY_KEYS.match(key):
            clean[key] = "<ID>"
        else:
            clean[key] = redact_text(value)
    return clean


def redact_event(event: Event) -> Event:
    """Redact an :class:`Event` in place (message, raw line, kv and multi-line block)."""
    event.message = redact_text(event.message)
    event.raw = redact_text(event.raw)
    event.kv = redact_kv(event.kv)
    if event.multiline:
        event.multiline = [redact_text(line) for line in event.multiline]
    return event
