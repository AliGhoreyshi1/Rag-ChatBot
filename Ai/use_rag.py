"""Compatibility API for the existing RAG interface.

Existing callers can continue using recommend_doctors and
recommend_doctors_json. A standalone fallback reads outputs/doctors_for_rag.csv
when the optional raw doctor CSV is not available.
"""

import json
import logging
from .rag_bridge import rag_recommend

logger = logging.getLogger(__name__)

__all__ = ["recommend_doctors", "recommend_doctors_json", "DOCTOR_TOOL_SCHEMA"]


def recommend_doctors(text: str) -> dict:
    if not isinstance(text, str):
        return {"query_type": "none", "doctors": [], "note": "input must be a string"}
    try:
        return rag_recommend(text)
    except Exception:
        logger.exception("doctor recommendation failed")
        return {"query_type": "error", "doctors": [], "note": "internal error while searching doctors"}


def recommend_doctors_json(text: str) -> str:
    return json.dumps(recommend_doctors(text), ensure_ascii=False)


DOCTOR_TOOL_SCHEMA = {
    "name": "recommend_doctors",
    "description": "Recommend up to three doctors by specialty or name from the doctor index.",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Doctor name or specialty and user preference."}},
        "required": ["text"],
    },
}
