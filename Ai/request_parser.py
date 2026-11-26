"""Conservative intent and entity extraction for the receptionist workflow."""

import re
from dataclasses import dataclass, field
from Services.rag_service import normalize_text
from .rag_bridge import doctors_table, supported_or_unknown


@dataclass
class ParsedRequest:
    doctor_names: list[str] = field(default_factory=list)
    specialties: list[str] = field(default_factory=list)
    unsupported_specialties: list[str] = field(default_factory=list)
    intent: str = "recommendation"


_DOCTOR_RE = re.compile(
    r"(?:دکتر|دكتر|dr\.?)(?:\s+)(.+?)"
    r"(?=\s+و\s+(?:دکتر|دكتر|dr\.?)\b|"
    r"\s+(?:متخصص|فوق\s+تخصص|برای|در|که|رو|را|از|بهتر|خوبه|چطوره|چطور|"
    r"هست|نیست|معرفی|پیشنهاد|نظرات|نظر|امتیاز|تعداد|تخصص|مطب|"
    r"نوبت|مقایسه|مقایسشون|مقایسهشون|چند|چندتا|چقدر|چیه|چیست|چی|میخوام|می\s+خوام|میخواهم|میشناسی|میدونی|بگو|برم|است)\b|[،؟?!.]|$)",
    flags=re.IGNORECASE,
)

_GENERIC_NAME_FIRST = {
    "خوب", "خوبی", "خوبه", "عالی", "بهترین", "مناسب", "متخصص", "خانم", "اقا", "زن", "مرد",
    "پوست", "ارتوپدی", "قلب", "مغز", "زنان", "کلیه", "گوش", "اطفال",
    "پیدا", "میخوام", "معرفی", "رو", "را", "ارزان", "نزدیک", "پر", "با", "که", "برای",
}


def extract_doctor_names(message: str) -> list[str]:
    text = normalize_text(message)
    matches = []
    for found in _DOCTOR_RE.finditer(text):
        fragment = found.group(1).strip()
        parts = fragment.split()
        if not parts or parts[0] in _GENERIC_NAME_FIRST:
            continue
        if len(parts) > 4:
            parts = parts[:4]
        candidate = "دکتر " + " ".join(parts)
        if candidate not in matches:
            matches.append(candidate)

    # Capture complete names mentioned without a 'doctor' prefix.
    # Only full matches present in the catalogue are accepted.
    try:
        available_names = doctors_table()["name"]
    except (FileNotFoundError, ValueError):
        available_names = []
    for name in available_names:
        normalized_name = normalize_text(str(name))
        bare = re.sub(r"^دکتر\s+", "", normalized_name).strip()
        if bare and len(bare.split()) >= 2:
            if re.search(r"(?<!\w)" + re.escape(bare) + r"(?!\w)", text):
                existing = [normalize_text(x) for x in matches]
                if normalized_name not in existing and not any(bare in x for x in existing):
                    matches.append(str(name))

    return matches


def parse_request(message: str) -> ParsedRequest:
    message = message.strip()
    specs, unsupported = supported_or_unknown(message)
    norm = normalize_text(message)

    if any(word in norm for word in ["مقایسه", "بین دکتر", "کدوم بهتر", "کدام بهتر"]):
        intent = "compare"
    elif any(word in norm for word in ["نوبت", "رزرو", "وقت ویزیت", "لغو نوبت"]):
        intent = "booking"
    elif any(word in norm for word in ["تعداد بیمار", "چند تا بیمار", "چند بیمار"]):
        intent = "patient_count"
    elif any(word in norm for word in ["نظر بیماران", "نظرات", "رضایت", "تجربه بیماران"]):
        intent = "reviews"
    elif any(word in norm for word in ["هزینه", "بیمه", "قیمت ویزیت", "تعرفه"]):
        intent = "pricing"
    elif any(word in norm for word in ["ادرس", "شماره تلفن", "تلفن مطب", "اطلاعات دکتر"]):
        intent = "doctor_info"
    else:
        intent = "recommendation"

    return ParsedRequest(
        doctor_names=extract_doctor_names(message),
        specialties=specs,
        unsupported_specialties=unsupported,
        intent=intent,
    )


_SYMPTOM_ONLY = (
    "سردرد", "سرگیجه", "درد زانو", "زانو درد", "کمردرد", "درد مفاصل",
    "ریزش مو", "گرفتگی بینی", "مشکل ادراری", "جوش صورت",
)
_DURATION_HINTS = (
    "روز", "هفته", "ماه", "سال", "ساعت", "دقیقه", "دیشب", "امروز",
    "دیروز", "صبح", "چند وقت", "مدتی", "مدت", "از وقتی", "از زمان",
)


def needs_symptom_detail(message: str) -> bool:
    """Ask one basic detail before matching an unspecific symptom to a doctor."""
    text = normalize_text(message)
    if extract_doctor_names(message):
        return False
    from .rag_bridge import detect_specialties
    if detect_specialties(message):
        return False
    if not any(term in text for term in _SYMPTOM_ONLY):
        return False
    if any(term in text for term in _DURATION_HINTS):
        return False
    return True
