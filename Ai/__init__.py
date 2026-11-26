"""AI integration layer for the existing doctor recommendation project."""

from .llm_service import MedicalLLM, generate_response

__all__ = ["MedicalLLM", "generate_response"]
