"""Deterministic high-priority routing rules; not clinical triage."""

import re
from Services.rag_service import normalize_text

# This is a conservative keyword screen, not a validated medical triage engine.
_EMERGENCY = (
    r"(?:درد|فشار)\s+شدید\s+(?:در\s+)?قفسه\s+سینه",
    r"درد\s+(?:شدید|خیلی\s+زیاد)\s+قلب",
    r"(?:نمی\s*تونم|نمی\s*توانم|نمیتونم)\s+نفس\s+بکشم",
    r"تنگی\s+نفس\s+شدید",
    r"نفس\s+کشیدن\s+(?:برام|برایم)\s+سخته",
    r"(?:ضعف|بی\s*حسی)\s+ناگهانی\s+(?:یک\s+طرف|نیمه)\s+بدن",
    r"اختلال\s+ناگهانی\s+(?:در\s+)?(?:گفتار|صحبت)",
    r"خونریزی\s+شدید",
    r"(?:از\s+هوش\s+رفته|بیهوش\s+شده)",
    r"تورم\s+(?:شدید\s+)?(?:زبان|گلو)",
    r"تشنج\s+(?:طولانی|مداوم)",
    r"سردرد\s+(?:بسیار\s+)?شدید\s+ناگهانی",
    r"بدترین\s+سردرد\s+زندگی",
)

_NEGATIVE = re.compile(r"^\s*(?:ندارم|ندارد|نداریم|نیست|نمی\s*باشد|نبود)")

_NONMEDICAL = (
    "اب و هوا", "هواشناسی", "پیش بینی هوا", "دما چند درجه", "بارون میاد",
    "قیمت دلار", "قیمت طلا", "بورس", "نتیجه فوتبال", "نتیجه بازی",
    "برنامه نویسی", "کد پایتون", "بازی کامپیوتری", "دستور پخت", "اشپزی",
    "فال حافظ", "شعر بگو", "تاریخ ایران", "ترجمه کن", "ریاضی حل کن",
    "شطرنج", "فوتبال", "رئیس جمهور", "رییس جمهور", "اخبار سیاسی",
    "خرید بلیط", "پرواز", "مسافرت", "تور مسافرتی", "فیلم سینمایی",
    "سریال", "موسیقی", "اهنگ", "اشپزی", "پخت کیک", "دانلود فیلم",
)

_MEDICAL = (
    "دکتر", "پزشک", "متخصص", "ویزیت", "مطب", "بیمار", "درمان", "دارو",
    "درد", "مرض", "بیماری", "سلامت", "پوست", "مغز", "اعصاب", "ارتوپد",
    "زنان", "زایمان", "کلیه", "گوش", "حلق", "بینی", "قلب", "دیابت",
    "زانو", "سردرد", "سرگیجه", "نفس", "عفونت", "تب", "جراحی", "نوبت",
    "نظر", "امتیاز", "تجربه", "حساسیت", "تیروئید", "اضطراب",
)

_MEDICATION = (
    "چه قرصی", "چه دارویی", "چه دوا", "دوز دارو", "دوزم رو", "دوز رو",
    "چند تا قرص", "چقدر دارو", "دارو تجویز", "قرص تجویز", "دوز رو بیشتر",
    "دارو رو قطع", "دارو را قطع", "مقدار مصرف", "چه مقدار مصرف",
)

_DIAGNOSIS = (
    "چه بیماری دارم", "چه مرضی دارم", "تشخیص بده", "بیماری من چیه",
    "به نظرت چه بیماری", "اسم بیماری من", "من چه بیماری",
)


def emergency_detected(message: str) -> bool:
    text = normalize_text(message)
    if "درد قفسه سینه" in text and "تنگی نفس" in text and not (
        "درد قفسه سینه ندارم" in text or "تنگی نفس ندارم" in text
    ):
        return True
    for expression in _EMERGENCY:
        match = re.search(expression, text)
        if match and not _NEGATIVE.search(text[match.end():match.end() + 16]):
            return True
    return False


def out_of_scope(message: str) -> bool:
    text = normalize_text(message)
    return any(word in text for word in _NONMEDICAL) and not any(word in text for word in _MEDICAL)


def medication_request(message: str) -> bool:
    text = normalize_text(message)
    return any(word in text for word in _MEDICATION)


def diagnosis_request(message: str) -> bool:
    text = normalize_text(message)
    return any(word in text for word in _DIAGNOSIS)


def medical_reply(message: str) -> str | None:
    if emergency_detected(message):
        return (
            "علائمی که گفتید ممکن است اورژانسی باشند. لطفاً منتظر دریافت نوبت نمانید "
            "و فوراً با اورژانس محلی تماس بگیرید یا برای دریافت کمک فوری اقدام کنید."
        )
    if out_of_scope(message):
        return (
            "من منشی سامانه معرفی پزشکان هستم و فقط می‌توانم درباره پزشکان، "
            "تخصص‌ها و خدمات پزشکی مرتبط با این سامانه کمک کنم."
        )
    if medication_request(message):
        return (
            "من امکان تجویز دارو یا تعیین و تغییر دوز دارو را ندارم. "
            "برای این موضوع با پزشک یا داروساز مشورت کنید. در صورت تمایل "
            "می‌توانم در پیدا کردن پزشک مناسب کمکتان کنم."
        )
    if diagnosis_request(message):
        return (
            "از روی این پیام نمی‌توانم بیماری را تشخیص بدهم. "
            "اگر علائم و مدت آن‌ها را توضیح بدهید، می‌توانم برای پیدا کردن "
            "تخصص مرتبط راهنمایی‌تان کنم؛ در صورت علائم شدید یا ناگهانی، فوری کمک پزشکی بگیرید."
        )
    return None
