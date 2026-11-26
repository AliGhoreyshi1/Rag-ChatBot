"""Run a simple command-line chat session using the integrated RAG + AvalAI."""

import sys
import uuid
from .chatbot import MedicalChatbot


def main() -> None:
    bot = MedicalChatbot()
    session_id = str(uuid.uuid4())
    print("Medical receptionist CLI. Type exit to quit.")
    while True:
        message = input("You: ").strip()
        if message.lower() in {"exit", "quit", "خروج"}:
            break
        try:
            print("Assistant:", bot.chat(message, session_id=session_id))
        except Exception as exc:
            print("Error:", str(exc), file=sys.stderr)


if __name__ == "__main__":
    main()
