"""Non-destructive adapter for the existing RAG implementation and output CSV.

The upstream RAG depends on Data/doctors_raw_canonical.csv, which is not bundled
with the uploaded archive. If it is available, the original recommender runs.
Otherwise, the adapter uses the already generated output CSV with compatible
specialty/aspect ranking logic. It never invents missing office or patient data.
"""

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from Services.data_service import DOCTORS_RAW_CANONICAL_CSV_PATH
from Services.rag_service import (
    ASPECTS,
    ASPECT_ALIASES,
    ASPECT_PRIOR_COUNT,
    ASPECT_WEIGHT,
    GENERAL_QUALITY_WORDS,
    NAME_MATCH_THRESHOLD,
    NAME_STOPWORDS,
    PRIOR_COUNT,
    SPECIALTY_GROUPS,
    TOP_N,
    normalize_text,
    name_tokens,
)
from .config import RAG_CSV_PATH


SPECIALTY_NAMES = (
    "مغز و اعصاب", "ارتوپدی", "زنان و زایمان", "گوش و حلق و بینی",
    "پوست و مو", "کلیه و مجاری ادراری",
)


def _word_present(text: str, alias: str) -> bool:
    alias = normalize_text(alias)
    return bool(alias and alias in text)


def detect_specialties(text: str) -> list[str]:
    normalized = normalize_text(text)
    found = []
    for index, group in enumerate(SPECIALTY_GROUPS):
        if any(_word_present(normalized, a) for a in group["aliases"]):
            found.append(SPECIALTY_NAMES[index])
    return found


# Symptom-to-specialty hints only determine which index to query, not diagnoses.
SYMPTOM_HINTS = {
    "سردرد": "مغز و اعصاب",
    "میگرن": "مغز و اعصاب",
    "تشنج": "مغز و اعصاب",
    "زانو درد": "ارتوپدی",
    "درد زانو": "ارتوپدی",
    "درد مفاصل": "ارتوپدی",
    "کمردرد": "ارتوپدی",
    "شکستگی": "ارتوپدی",
    "اکنه": "پوست و مو",
    "جوش صورت": "پوست و مو",
    "ریزش مو": "پوست و مو",
    "بیماری پوستی": "پوست و مو",
    "مشکل ادراری": "کلیه و مجاری ادراری",
    "سنگ کلیه": "کلیه و مجاری ادراری",
    "نازایی": "زنان و زایمان",
    "بارداری": "زنان و زایمان",
    "درد گوش": "گوش و حلق و بینی",
    "قاعدگی": "زنان و زایمان",
    "عفونت گوش": "گوش و حلق و بینی",
    "گرفتگی بینی": "گوش و حلق و بینی",
}

# These cases are familiar medical conditions but absent from the six index groups.
UNSUPPORTED_HINTS = {
    "دیابت": "غدد و متابولیسم",
    "غدد": "غدد و متابولیسم",
    "تیروئید": "غدد و متابولیسم",
    "قند خون": "غدد و متابولیسم",
    "بیماری قلبی": "قلب و عروق",
    "قلب": "قلب و عروق",
    "درد قلب": "قلب و عروق",
    "تپش قلب": "قلب و عروق",
    "دندان": "دندان‌پزشکی",
    "چشم پزشک": "چشم‌پزشکی",
    "چشم": "چشم‌پزشکی",
    "گوارش": "گوارش",
    "معده درد": "گوارش",
    "کبد": "گوارش",
    "هپاتیت": "گوارش",
    "بیماری اسم": "ریه",
    "تنگی نفس مزمن": "ریه",
    "سرطان": "انکولوژی",
    "کم خونی": "داخلی / خون",
    "فشار خون": "داخلی / قلب و عروق",
    "روان پزشک": "روان‌پزشکی",
    "روانپزشک": "روان‌پزشکی",
}


def supported_or_unknown(text: str) -> tuple[list[str], list[str]]:
    specialties = detect_specialties(text)
    normalized = normalize_text(text)
    unsupported = []
    for alias, title in UNSUPPORTED_HINTS.items():
        if _word_present(normalized, alias) and title not in unsupported:
            unsupported.append(title)
    if not specialties:
        for alias, specialty in SYMPTOM_HINTS.items():
            if _word_present(normalized, alias) and specialty not in specialties:
                specialties.append(specialty)
    return specialties, unsupported


@lru_cache(maxsize=1)
def doctors_table() -> pd.DataFrame:
    if not RAG_CSV_PATH.exists():
        raise FileNotFoundError(f"RAG output file not found: {RAG_CSV_PATH}")
    df = pd.read_csv(RAG_CSV_PATH, encoding="utf-8-sig")
    required = {"doctor_id", "name", "specialty", "comment_count", "comment_rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RAG output missing columns: {', '.join(sorted(missing))}")
    df["doctor_id"] = df["doctor_id"].astype(str)
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0)
    df["comment_rating"] = pd.to_numeric(df["comment_rating"], errors="coerce")
    df["_name_norm"] = df["name"].fillna("").map(normalize_text)
    df["_spec_norm"] = df["specialty"].fillna("").map(normalize_text)
    return df


def find_doctor(name: str) -> dict[str, Any] | None:
    df = doctors_table()
    norm = normalize_text(name)
    tokens = name_tokens(name)
    if not tokens:
        return None
    # Exact token-matching avoids identifying a different doctor by mistake.
    candidates = df.loc[df["_name_norm"].map(lambda s: s == norm or name_tokens(s) == tokens)]
    if len(candidates) != 1:
        return None
    return _format_record(candidates.iloc[0])


def _safe_num(value: Any) -> float | int | None:
    try:
        if pd.isna(value):
            return None
        n = float(value)
        if not math.isfinite(n):
            return None
        return int(n) if n.is_integer() else n
    except (TypeError, ValueError):
        return None


def _format_record(row: pd.Series) -> dict[str, Any]:
    output = {
        "doctor_id": str(row["doctor_id"]),
        "name": str(row["name"]),
        "specialty": str(row["specialty"]),
        "rating": _safe_num(row.get("comment_rating")),
        "program_rating": _safe_num(row.get("program_rating")),
        "comment_count": _safe_num(row.get("comment_count")),
        "profile_url": str(row["profile_url"]) if pd.notna(row.get("profile_url")) else None,
        "offices": [],
    }
    scores = {}
    for key, (value_col, count_col) in ASPECTS.items():
        val = _safe_num(row.get(value_col))
        count = _safe_num(row.get(count_col))
        if val is not None and count:
            scores[key] = {"score": val, "comment_count": count}
    if scores:
        output["aspects"] = scores
    return output


def _aspect_keys(text: str) -> list[str]:
    norm = normalize_text(text)
    found = []
    for key, aliases in ASPECT_ALIASES.items():
        if any(normalize_text(a) in norm for a in aliases):
            found.append(key)
    if not found and any(normalize_text(w) in norm for w in GENERAL_QUALITY_WORDS):
        found.append("quality")
    return found


def _specialty_pool(df: pd.DataFrame, specialty: str) -> pd.DataFrame:
    query = normalize_text(specialty)
    keywords = []
    for group in SPECIALTY_GROUPS:
        if any(normalize_text(a) in query for a in group["aliases"]):
            keywords += [normalize_text(k) for k in group["keywords"]]
    if not keywords:
        keywords = [query]
    return df.loc[df["_spec_norm"].map(lambda x: any(k in x for k in keywords))].copy()


def _score_pool(df: pd.DataFrame, preferences: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    mean = float(doctors_table()["comment_rating"].mean())
    if not math.isfinite(mean):
        mean = 0.0
    n = out["comment_count"]
    rating = out["comment_rating"].fillna(mean)
    score = (n / (n + PRIOR_COUNT)) * rating + (PRIOR_COUNT / (n + PRIOR_COUNT)) * mean
    aspect_keys = _aspect_keys(preferences)
    available_aspects = []
    for key in aspect_keys:
        value_col, count_col = ASPECTS[key]
        if value_col not in out.columns or count_col not in out.columns:
            continue
        all_values = pd.to_numeric(doctors_table()[value_col], errors="coerce")
        aspect_mean = all_values.mean()
        if pd.isna(aspect_mean):
            continue
        observed = pd.to_numeric(out[value_col], errors="coerce").fillna(aspect_mean)
        counts = pd.to_numeric(out[count_col], errors="coerce").fillna(0)
        available_aspects.append((counts / (counts + ASPECT_PRIOR_COUNT)) * observed + (ASPECT_PRIOR_COUNT / (counts + ASPECT_PRIOR_COUNT)) * aspect_mean)
    if available_aspects:
        avg = sum(available_aspects) / len(available_aspects)
        score = ASPECT_WEIGHT * avg + (1.0 - ASPECT_WEIGHT) * score
    out["_rank_score"] = score
    return out.sort_values(["_rank_score", "comment_count"], ascending=False)


def fallback_recommend(text: str, specialty: str | None = None, doctor_name: str | None = None) -> dict[str, Any]:
    df = doctors_table()
    if doctor_name:
        doctor = find_doctor(doctor_name)
        if doctor:
            return {"query_type": "name", "doctors": [doctor]}
        return {"query_type": "none", "doctors": [], "note": "doctor not found"}
    requested = specialty or text
    pool = _specialty_pool(df, requested)
    pool = _score_pool(pool, text)
    if pool.empty:
        return {"query_type": "none", "doctors": [], "note": "no matching specialty"}
    return {
        "query_type": "specialty", "doctors": [
            _format_record(r) for _, r in pool.head(TOP_N).iterrows()
        ],
    }


def _join_metadata(doctor: dict[str, Any]) -> dict[str, Any]:
    record = find_doctor(str(doctor.get("name", "")))
    if record:
        record["offices"] = doctor.get("offices") or []
        return record
    return doctor


def rag_recommend(text: str, specialty: str | None = None, doctor_name: str | None = None) -> dict[str, Any]:
    """Prefer existing RAG when its complete dataset is present."""
    if DOCTORS_RAW_CANONICAL_CSV_PATH and Path(DOCTORS_RAW_CANONICAL_CSV_PATH).is_file():
        try:
            from Services.rag_service import _get_bot
            data = _get_bot().recommend(specialty=specialty, doctor_name=doctor_name, aspects=[text])
            data["doctors"] = [_join_metadata(item) for item in data.get("doctors", [])]
            return data
        except (OSError, ValueError, KeyError):
            pass
    return fallback_recommend(text, specialty=specialty, doctor_name=doctor_name)
