"""AvalAI credentials and runtime configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


_load_environment()

AVALAI_BASE_URL = os.getenv("AVALAI_BASE_URL", "https://api.avalai.ir/v1").strip()
AVALAI_MODEL = os.getenv("AVALAI_MODEL", "gpt-5.6-sol").strip()
AVALAI_TIMEOUT_SECONDS = float(os.getenv("AVALAI_TIMEOUT_SECONDS", "45"))
AVALAI_MAX_TOKENS = int(os.getenv("AVALAI_MAX_TOKENS", "1800"))
RAG_CSV_PATH = PROJECT_ROOT / "outputs" / "doctors_for_rag.csv"


def get_api_key() -> str:
    key = os.getenv("AVALAI_API_KEY", "").strip()
    if not key or key == "YOUR_AVALAI_API_KEY":
        raise RuntimeError("AVALAI_API_KEY is missing. Copy .env.example to .env and set the key.")
    return key
