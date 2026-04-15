"""
Response Generator + Response Validator
Uses an LLM (via Anthropic SDK or OpenAI) to synthesize retrieved context into a final answer.
Response Validator performs a post-generation safety + completeness check.
"""

import os
import re
from dataclasses import dataclass
from query_router import RetrievedDoc
from query_expander import ExpandedQuery


# --- Response Generator -------------------------------------------------------

@dataclass
class GeneratedResponse:
    answer: str
    sources_used: list[str]
    raw_context: str


def _build_context(docs: list[RetrievedDoc]) -> str:
    """Formats retrieved docs into a numbered context block for the prompt."""
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] ({doc.source}) {doc.text}")
    return "\n".join(lines)


def _build_prompt(query: str, context: str) -> str:
    return f"""You are a clinical pharmacist assistant. 
Answer the patient's question using ONLY the provided context.
If the context is insufficient, say so clearly. Never invent drug information.

CONTEXT:
{context}

PATIENT QUESTION: {query}

Respond concisely (2-4 sentences). Mention relevant drug names, dosages, and any critical warnings."""


class ResponseGenerator:
    """
    Calls an LLM to synthesize retrieved context into a final response.
    Supports: Anthropic Claude (default), OpenAI, or a stub fallback for testing.
    """

    def __init__(self, provider: str = "anthropic", model: str = "claude-opus-4-5"):
        self.provider = provider
        self.model    = model

    def generate(self, expanded: ExpandedQuery, docs: list[RetrievedDoc]) -> GeneratedResponse:
        context    = _build_context(docs)
        prompt     = _build_prompt(expanded.original, context)
        source_ids = [d.id for d in docs]

        answer = self._call_llm(prompt)

        return GeneratedResponse(
            answer=answer,
            sources_used=source_ids,
            raw_context=context,
        )

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "openrouter":
            return self._call_openrouter(prompt)
        elif self.provider == "google":
            return self._call_google(prompt)
        else:
            return self._stub_response(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"[LLM Error - Anthropic] {e}"

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM Error - OpenAI] {e}"

    def _call_openrouter(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
            response = client.chat.completions.create(
                model=self.model if "models/" in self.model else f"openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM Error - OpenRouter] {e}"

    def _call_google(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            # Using Gemini 1.5 Pro as the 'MedTech' upgraded model
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[LLM Error - Google Gemini] {e}"

    def _stub_response(self, prompt: str) -> str:
        """Deterministic stub — useful for unit tests without API keys."""
        return (
            "Based on the available information, the recommended treatment appears in "
            "the context provided. Please consult a licensed pharmacist before use. "
            "[STUB RESPONSE - set ANTHROPIC_API_KEY or OPENAI_API_KEY for real answers]"
        )


# --- Response Validator -------------------------------------------------------

SAFETY_KEYWORDS = [
    "consult", "doctor", "pharmacist", "warning", "contraindicated",
    "side effect", "overdose", "seek medical", "do not", "avoid",
]

HALLUCINATION_SIGNALS = [
    r'\b(?:always|never|guaranteed|100%|cure)\b',
    r'\b(?:clinical trial|study shows|research proves)\b',
]


@dataclass
class ValidationResult:
    passed: bool
    warnings: list[str]
    final_answer: str


class ResponseValidator:
    """
    Post-generation safety and completeness checker.
    Flags potentially dangerous or incomplete responses.
    """

    MIN_ANSWER_LENGTH = 40

    def validate(self, response: GeneratedResponse, original_query: str) -> ValidationResult:
        warnings = []

        # 1. Length check
        if len(response.answer) < self.MIN_ANSWER_LENGTH:
            warnings.append("Response is too short — may be incomplete.")

        # 2. Hallucination signal check
        for pattern in HALLUCINATION_SIGNALS:
            if re.search(pattern, response.answer, re.IGNORECASE):
                warnings.append(f"Potential overconfidence detected: pattern '{pattern}'")

        # 3. Safety disclaimer check — add if missing
        has_safety_note = any(kw in response.answer.lower() for kw in SAFETY_KEYWORDS)
        final_answer = response.answer
        if not has_safety_note:
            final_answer += "\n\n[WARNING] Always consult a licensed pharmacist or physician before taking any medication."
            warnings.append("Safety disclaimer auto-appended.")

        # 4. Source coverage check
        if not response.sources_used:
            warnings.append("No sources were cited in this response.")

        passed = len([w for w in warnings if "overconfidence" in w or "too short" in w]) == 0

        return ValidationResult(
            passed=passed,
            warnings=warnings,
            final_answer=final_answer,
        )
