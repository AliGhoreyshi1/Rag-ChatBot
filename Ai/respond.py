"""CLI entry point for the direct user message + RAG JSON -> answer contract."""

import argparse
from pathlib import Path
from .llm_service import generate_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Persian reply from a message and a RAG JSON file.")
    parser.add_argument("--message", required=True, help="User message in Persian")
    parser.add_argument("--rag-file", required=True, type=Path, help="RAG output .json (empty files are supported)")
    args = parser.parse_args()
    rag_json = args.rag_file.read_text(encoding="utf-8-sig")
    print(generate_response(args.message, rag_json))


if __name__ == "__main__":
    main()
