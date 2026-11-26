"""Orchestrator: user message -> single/multiple RAG lookups -> LLM string."""

import json
from threading import RLock
from typing import Any

from .llm_service import MedicalLLM
from .rag_bridge import detect_specialties, find_doctor, rag_recommend
from .request_parser import ParsedRequest, needs_symptom_detail, parse_request
from .safety import medical_reply


class MedicalChatbot:
    """Development session state is in-memory; use persistent storage in production."""

    def __init__(self, llm: MedicalLLM | None = None):
        self.llm = llm or MedicalLLM()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def _retrieve(self, message: str, parsed: ParsedRequest) -> dict[str, Any]:
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        sources = []

        if parsed.doctor_names:
            for name in parsed.doctor_names:
                doctor = find_doctor(name)
                if doctor is None:
                    sources.append({"query": name, "query_type": "name", "status": "not_found"})
                    continue
                if parsed.specialties and not any(
                    spec in detect_specialties(doctor["specialty"]) for spec in parsed.specialties
                ):
                    sources.append({"query": name, "query_type": "name", "status": "specialty_mismatch"})
                    continue
                result = rag_recommend(message, doctor_name=name)
                sources.append({"query": name, "query_type": "name", "status": result.get("query_type")})
                for doc in result.get("doctors", []):
                    if doc.get("name") != doctor["name"]:
                        continue
                    key = str(doc.get("doctor_id") or doc["name"])
                    if key not in seen:
                        seen.add(key)
                        found.append(doc)
        else:
            for specialty in parsed.specialties:
                result = rag_recommend(message, specialty=specialty)
                sources.append({"query": specialty, "query_type": "specialty", "status": result.get("query_type")})
                for doc in result.get("doctors", []):
                    key = str(doc.get("doctor_id") or doc["name"])
                    if key not in seen:
                        seen.add(key)
                        found.append(doc)

        return {"query_type": "combined", "doctors": found, "sources": sources}

    def chat(
        self,
        user_message: str,
        session_id: str | None = None,
        rag_json: str | dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string")
        current = user_message.strip()
        if not current:
            return "لطفاً درخواست‌تان را بنویسید."

        direct = medical_reply(current)
        if direct:
            return direct

        history = []
        context_query = current
        if session_id:
            with self._lock:
                previous = self._sessions.get(session_id, {})
            explicit_new_topic = any(
                keyword in current for keyword in ["موضوع جدید", "بحث دیگه", "موضوع دیگه"]
            )
            reference_to_prior = any(
                phrase in current for phrase in [
                    "همون", "اون دکتر", "این دکتر", "امتیازش", "مطبش",
                    "نظراتش", "آدرسش", "ادرسش", "بیمه اش", "کدومشون",
                    "تعداد نظرات", "تعداد بیمارها", "حالا امتیاز", "هزینه اش",
                ]
            )
            if previous.get("query") and not explicit_new_topic and (
                previous.get("awaiting_detail") or reference_to_prior
            ):
                context_query = previous["query"] + " ؛ " + current
                history = previous.get("history", [])

        parsed = parse_request(context_query)
        if needs_symptom_detail(context_query):
            answer = (
                "این علائم از چه زمانی شروع شده‌اند و آیا ناگهانی یا شدید هستند؟ "
                "با این اطلاعات می‌توانم برای پیدا کردن پزشک مناسب راهنمایی کنم."
            )
            self._save(session_id, context_query, True, history, current, answer)
            return answer
        if rag_json is not None:
            result = rag_json
        else:
            if not parsed.specialties and not parsed.doctor_names and not parsed.unsupported_specialties:
                if parsed.intent == "booking":
                    return "در حال حاضر سیستم نوبت‌دهی متصل نیست؛ می‌توانم پزشک مرتبط را معرفی کنم."
                if parsed.intent in {"reviews", "patient_count", "doctor_info", "pricing", "compare"}:
                    answer = "نام پزشک یا تخصص موردنظر را بگویید تا اطلاعات موجود را بررسی کنم."
                else:
                    answer = "برای پیدا کردن پزشک مناسب، لطفاً مشکل یا تخصص موردنظر و در صورت نیاز مدت علائم را بگویید."
                self._save(session_id, context_query, True, history, current, answer)
                return answer
            result = self._retrieve(context_query, parsed)
            mismatches = [
                s["query"] for s in result["sources"]
                if s.get("status") == "specialty_mismatch"
            ]
            if mismatches and not result["doctors"]:
                answer = (
                    "نام پزشک در فهرست ما وجود دارد، اما تخصص ثبت‌شده او با تخصصی "
                    "که درخواست کردید مطابقت ندارد. لطفاً نام یا تخصص را بررسی کنید."
                )
                self._save(session_id, context_query, False, history, current, answer)
                return answer

        answer = self.llm.generate_response(context_query, result, history=history)
        self._save(session_id, context_query, False, history, current, answer)
        return answer

    def _save(
        self,
        session_id: str | None,
        query: str,
        awaiting_detail: bool,
        history: list[dict[str, str]],
        user_message: str,
        answer: str,
    ) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions[session_id] = {
                "query": query,
                "awaiting_detail": awaiting_detail,
                "history": (history + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": answer},
                ])[-4:],
            }


def chat(user_message: str, session_id: str | None = None) -> str:
    """Convenience one-liner with an in-process default bot."""
    return _DEFAULT_BOT.chat(user_message, session_id=session_id)


_DEFAULT_BOT = MedicalChatbot()
