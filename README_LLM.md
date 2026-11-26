# AvalAI LLM Integration for the Existing RAG-Chatbot

The original training/data pipelines and `Services/` package are preserved. The new conversational LLM layer lives in the **existing `Ai/` directory** (capital `A`, as named in the uploaded project). All new code comments and docstrings are in English.

## Architecture

```text
User message ----------------------------------------+
     |                                               |
     v                                               |
Ai/safety.py (emergency, out-of-scope, medical)       |
     |                                               |
     v                                               |
Ai/request_parser.py (names, specialties, intent)    |
     |                                               |
     +--> Missing symptom details? -> Follow-up      |
     |                                               |
     v                                               |
Ai/chatbot.py (request orchestration, session context)|
     |                                               |
     v                                               |
Ai/rag_bridge.py -- multiple names/specialties       |
     |                                               |
     +--> Services/rag_service.py (when raw CSV exists)
     |         OR                                    |
     +--> outputs/doctors_for_rag.csv (CSV fallback)  |
     |                                               |
     v                                               |
RAG result JSON --------------------------------------+ 
     |
     v
Ai/llm_service.py (coverage checks + grounded system prompt)
     |
     +--> AvalAI chat/completions only if needed
     |
     v
Persian string reply
```

**Important:** `Ai/llm_service.py` also works as a standalone component if your upstream system has already produced the RAG JSON. In that case it **does not call RAG again**.

## Setup (Python 3.11 recommended)

On Windows PowerShell, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ai.txt
Copy-Item .env.example .env
```

Set your private API key in `.env`:

```env
AVALAI_API_KEY=YOUR_AVALAI_API_KEY
AVALAI_BASE_URL=https://api.avalai.ir/v1
AVALAI_MODEL=gpt-5.6-sol
AVALAI_MAX_TOKENS=1800
AVALAI_TIMEOUT_SECONDS=45
```

Use a model ID enabled for your AvalAI account. This program uses OpenAI-compatible **Chat Completions**. The key is never hardcoded or included in the archive.

## A. Your requested direct interface: user message + RAG JSON -> string

```python
import json
from Ai.llm_service import generate_response

user_message = "متخصص پوست می‌خواهم"
rag_result = {
    "query_type": "specialty",
    "doctors": [
        {
            "name": "EXAMPLE DOCTOR (NOT REAL DATA)",
            "specialty": "متخصص پوست و مو",
            "rating": 9.2,
            "comment_count": 12,
            "offices": []
        }
    ]
}

answer: str = generate_response(
    user_message=user_message,
    rag_json=json.dumps(rag_result, ensure_ascii=False)
)
print(answer)
```

If RAG produces a JSON file on disk:

```python
from pathlib import Path
from Ai.llm_service import generate_response

rag_json = Path("rag_result.json").read_text(encoding="utf-8-sig")
response: str = generate_response("یه متخصص پوست می‌خوام", rag_json)
print(response)
```

If RAG returns an empty file, an empty string, `None`, `{"doctors":[]}` or `{"query_type":"none","doctors":[]}`, the code returns a useful fallback **without calling AvalAI**.

CLI variant:

```powershell
python -m Ai.respond --message "متخصص پوست می‌خوام" --rag-file path\to\rag_output.json
```

## B. Integrated interface: user message -> RAG -> LLM -> string

```python
from Ai.chatbot import MedicalChatbot

bot = MedicalChatbot()

print(bot.chat("متخصص پوست و ارتوپدی می‌خوام", session_id="demo-user-1"))
print(bot.chat("دکتر وحید مجریان و دکتر پویا مشکدار رو مقایسه کن", session_id="demo-user-2"))
print(bot.chat("یه دکتر خوب می‌خوام", session_id="demo-user-3"))
print(bot.chat("پوست", session_id="demo-user-3"))
```

Use one `MedicalChatbot` instance for multiple requests if you want in-memory conversation context. The `session_id` should be stable for a user conversation. Production deployments should use authenticated, access-controlled persistent storage instead of in-memory sessions.

Interactive CLI:

```powershell
python -m Ai.demo
```

## RAG output schema

The existing `Services/rag_service.py` returns a dictionary like:

```json
{
  "query_type": "specialty",
  "doctors": [
    {
      "name": "Example doctor",
      "specialty": "متخصص پوست و مو",
      "rating": 9.1,
      "offices": []
    }
  ]
}
```

The adapter can enrich doctors using the real data in `outputs/doctors_for_rag.csv` with `doctor_id`, `profile_url`, `comment_count`, `program_rating` and available aggregate aspect scores. It does **not** fabricate the missing addresses, raw reviews or patient counts.

**Data interpretation:** `comment_count` is number of comments, not patient count; `comment_rating`/`program_rating` in this repository use a 0–10 scale. The CSV does not have actual patient counts or raw comment texts. The LLM should acknowledge these limits rather than inventing data.

## Handled scenarios

| Scenario | Behavior |
|---|---|
| Generic "recommend a good doctor" | Ask for the specialty or relevant symptoms |
| Symptom without duration | Ask when symptoms began and whether sudden/severe |
| Direct specialty request | Retrieve doctors for that specialty |
| Unknown doctor name + valid specialty | Say the named doctor is not listed; never silently substitute someone else |
| Doctor name with wrong specialty | Report specialty mismatch |
| Several specialties | Call RAG separately per specialty and merge unique doctors |
| Several named doctors | Call RAG separately per doctor, validate names, report missing doctors |
| Disease not represented in the corpus | Explain that no doctors for the inferred specialty are in the current data |
| Out-of-scope request (e.g. weather) | Explain that the bot handles doctor/medical-service requests only |
| RAG empty output or no matching result | Return a follow-up or no-result message without making up doctors |
| Reviews and patient counts | Use only available records; distinguish comments from patients |
| Fees/insurance not available | Say the information is missing |
| Emergency red flags | Immediate emergency guidance, without invoking RAG or LLM |
| Medication and definitive diagnosis requests | Do not prescribe or diagnose |
| Booking requests | Clearly state that real booking is not connected |

## Important limitations from the uploaded ZIP

1. `Ai/use_rag.py` in the archive imported `Ai.rag_service`, but the actual recommender lives at `Services/rag_service.py`. The new `Ai/use_rag.py` retains the same public functions and routes them to `Ai/rag_bridge.py`.
2. `Services/rag_service.py` needs `Data/doctors_raw_canonical.csv` in addition to `outputs/doctors_for_rag.csv`. That raw CSV is **not present in the uploaded archive**. The bridge therefore uses a fallback reading the existing output CSV until the raw CSV is supplied, at which point it prefers the existing recommender logic.
3. Some of the pre-existing tests are out of sync with the implementation: `Tests/test_rag_service.py` expects `build_doctor_frame` and `prepare_comments` in `Services/rag_service.py`, but those functions are missing from that file in the provided ZIP. `Tests/test_plot_service.py` requires extra plotting packages (`arabic_reshaper`, `python-bidi`). None of the original Services or pipeline files were edited to mask these issues.
4. The safety check is a basic deterministic keyword filter, **not a medically validated triage system**. Real clinical use requires clinician review, stronger detection and escalation policies.
5. An authentic AvalAI request requires your API key and network access; offline tests use a fake LLM to verify routing and prompt construction.
6. The fallback index only supports the six speciality categories configured in the RAG source: neurology, orthopedics, gynecology/obstetrics, ENT, dermatology, and urology. It does not imply that all diseases are covered. Broad or ambiguous symptoms may require clarification.

## Tests

AI integration tests (offline):

```powershell
python -m pytest Tests/test_ai_integration.py -q
```

Original tests excluding the two unrelated pre-existing collection issues:

```powershell
python -m pytest -q --ignore=Tests/test_plot_service.py --ignore=Tests/test_rag_service.py
```

## Production/security notes

Use HTTPS, authenticated user accounts, API rate limits, limited log retention, and appropriate encryption/access control for health data. Avoid storing raw health details unnecessarily. Avoid sending identifiers that are not needed to the provider. The existing `session_id` mechanism is suitable only for local development. The LLM must not be used for autonomous medical triage, diagnosis, prescribing or confirmed appointment booking.
