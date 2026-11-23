import json
import logging


from Ai.rag_service import get_recommendations as _core_recommend

logger = logging.getLogger(__name__)

__all__ = [
    "recommend_doctors",
    "recommend_doctors_json",
    "DOCTOR_TOOL_SCHEMA",
]


def recommend_doctors(text: str) -> dict:
    if not isinstance(text, str):
        return {"query_type": "none", "doctors": [], "note": "input must be a string"}

    try:
        return _core_recommend(text)
    except Exception:                                   
        logger.exception("doctor recommendation failed")
        return {"query_type": "error", "doctors": [],
                "note": "internal error while searching doctors"}


def recommend_doctors_json(text: str) -> str:
    return json.dumps(recommend_doctors(text), ensure_ascii=False)


DOCTOR_TOOL_SCHEMA = {
    "name": "recommend_doctors",
    "description": (
        "پیشنهاد حداکثر ۳ دکتر برتر بر اساس تخصص و ترجیح کاربر. "
        "تخصص‌های موجود: مغز و اعصاب، ارتوپدی، زنان و زایمان، گوش و حلق و بینی، پوست، کلیه و مجاری ادراری. "
        "ترجیح‌های قابل تشخیص: انتظار/معطلی، هزینه، رفتار، نتیجهٔ درمان، کارکنان، کیفیت کلی. "
        "اگر دکتری پیدا نشد، لیست doctors خالی است."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "تخصص و ترجیح کاربر، مثل «ارتوپدی معطل نکنه»، یا اسم دکتر",
            }
        },
        "required": ["text"],
    },
}

# from Services.doctor_service import recommend_doctors, recommend_doctors_json

# result = recommend_doctors("ارتوپدی معطل نکنه")        # دیکشنری
# text = recommend_doctors_json("ارتوپدی معطل نکنه")      # رشتهٔ JSON