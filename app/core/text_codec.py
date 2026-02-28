from __future__ import annotations

from typing import Any


MOJIBAKE_MARKERS = ("Ã", "Â", "æ", "ç", "å", "ä", "é", "è", "ê", "ô", "ö", "ï", "ð")


class EncodingNormalizationError(ValueError):
    def __init__(self, *, field_path: str, sample: str, message: str = "invalid text encoding") -> None:
        self.field_path = field_path
        self.sample = sample[:80]
        self.message = message
        super().__init__(f"{message}: {field_path}")

    def to_error_detail(self) -> dict[str, str]:
        return {
            "error_code": "invalid_text_encoding",
            "field": self.field_path,
            "sample": self.sample,
            "message": self.message,
        }


def _control_char_count(text: str) -> int:
    return sum(1 for ch in text if 0x80 <= ord(ch) <= 0x9F)


def _quality_score(text: str) -> float:
    cjk_count = sum(1 for ch in text if "一" <= ch <= "龿")
    printable_count = sum(1 for ch in text if ch.isprintable())
    marker_hits = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    replacement_count = text.count("�")
    return (
        (cjk_count * 6.0)
        + (printable_count * 0.05)
        - (_control_char_count(text) * 8.0)
        - (replacement_count * 12.0)
        - (marker_hits * 2.0)
    )


def _looks_mojibake(text: str) -> bool:
    if not text:
        return False
    if "�" in text:
        return True
    if _control_char_count(text) > 0:
        return True
    marker_hits = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return marker_hits >= 2


def _try_decode(text: str, source_encoding: str) -> str | None:
    try:
        raw = text.encode(source_encoding, errors="strict")
    except UnicodeEncodeError:
        return None
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def _repair_text(text: str) -> str | None:
    best = text
    best_score = _quality_score(text)
    for source_encoding in ("latin-1", "cp1252"):
        candidate = _try_decode(text, source_encoding)
        if not candidate or candidate == text:
            continue
        score = _quality_score(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    return best if best != text else None


def normalize_text(value: str, *, field_path: str = "text", strict: bool = True) -> str:
    if not isinstance(value, str):
        return value
    if not _looks_mojibake(value):
        return value

    repaired = _repair_text(value)
    if repaired is not None and _quality_score(repaired) > _quality_score(value):
        return repaired

    if strict:
        raise EncodingNormalizationError(field_path=field_path, sample=value)
    return value


def normalize_payload(payload: Any, *, field_path: str = "payload", strict: bool = True) -> Any:
    if isinstance(payload, str):
        return normalize_text(payload, field_path=field_path, strict=strict)
    if isinstance(payload, list):
        return [
            normalize_payload(item, field_path=f"{field_path}[{index}]", strict=strict)
            for index, item in enumerate(payload)
        ]
    if isinstance(payload, tuple):
        return tuple(
            normalize_payload(item, field_path=f"{field_path}[{index}]", strict=strict)
            for index, item in enumerate(payload)
        )
    if isinstance(payload, set):
        return {
            normalize_payload(item, field_path=f"{field_path}[{index}]", strict=strict)
            for index, item in enumerate(payload)
        }
    if isinstance(payload, dict):
        normalized: dict[Any, Any] = {}
        for key, value in payload.items():
            normalized[key] = normalize_payload(value, field_path=f"{field_path}.{key}", strict=strict)
        return normalized
    return payload


def normalize_dict(values: dict[str, Any], *, field_path: str = "payload", strict: bool = True) -> dict[str, Any]:
    normalized = normalize_payload(values, field_path=field_path, strict=strict)
    return normalized if isinstance(normalized, dict) else {}
