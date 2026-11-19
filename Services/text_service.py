"""نرمال‌سازی متن فارسی و تقسیم به جمله."""
import re

import pandas as pd

from Services.keywords import SPACING_MAP

try:
    from hazm import Normalizer
    _HAZM = Normalizer()
except ImportError:
    _HAZM = None

ELONGATION = re.compile(r"(.)\1{2,}")
MULTI_SPACE = re.compile(r"\s+")
SENTENCE_SPLIT = re.compile(r"[.!؟?\n]+")

_PUNCT = r"\s.،!؟?,;:()«»\"'"
_SPACING_PATTERNS = [
    (re.compile(rf"(?<![^{_PUNCT}]){re.escape(src)}(?![^{_PUNCT}])"), dst)
    for src, dst in sorted(SPACING_MAP.items(), key=lambda kv: -len(kv[0]))
]


def normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""
    if _HAZM is not None:
        text = _HAZM.normalize(text)
    text = text.replace("\u200c", " ")
    text = ELONGATION.sub(r"\1", text)
    for pattern, replacement in _SPACING_PATTERNS:
        text = pattern.sub(replacement, text)
    return MULTI_SPACE.sub(" ", text).strip()


def normalize_column(df: pd.DataFrame, column: str = "text", output_column: str = "text_norm") -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Missing text column: {column}")
    out = df.copy()
    out[output_column] = out[column].apply(normalize_text)
    return out


def split_sentences(text) -> list[str]:
    if not isinstance(text, str):
        return []
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]