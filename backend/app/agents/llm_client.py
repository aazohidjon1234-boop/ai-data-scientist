"""Optional LLM client (OpenAI-compatible API) for richer interpretation.

The LLM NEVER computes anything. It receives the JSON produced by the
Python tools and is instructed to only use those numbers. If no API key is
configured (or the call fails) the deterministic local explanation engine
is used instead — the application always works.

Set in the environment (never in code or the frontend):
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from ..core.config import get_settings

SYSTEM_PROMPT = (
    "You are a senior data scientist explaining results to a business user. "
    "Rules: (1) You may ONLY cite numbers that appear in the JSON results provided. "
    "Never invent, round into, or estimate metrics. (2) Use simple language and short paragraphs. "
    "(3) Be honest about limitations. (4) Format with Markdown headings, bold and bullet lists. "
    "If a number you want to mention is not in the JSON, do not mention it."
)


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def complete(self, user_prompt: str, max_tokens: int = 1400, temperature: float = 0.2) -> str | None:
        if not self.enabled:
            return None
        try:
            resp = httpx.post(
                f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                json={
                    "model": self.settings.llm_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=90.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip() or None
        except Exception:
            return None  # fall back to the local engine


def build_interpretation_prompt(kind: str, results: dict[str, Any], question: str | None = None) -> str:
    payload = json.dumps(results, default=str)[:60_000]
    if kind == "analysis":
        task = "Explain this dataset analysis in plain language: what the data contains, data quality issues, how the problem type was decided, and what a user should keep in mind."
    elif kind == "training":
        task = "Explain this model training run in plain language: what was prepared, how the models compared, why the best model won, what the key metrics mean, and how to improve."
    else:
        task = f"Answer the user's question about this analysis/training. User question: {question}"
    return f"{task}\n\nResults JSON (the ONLY source of numbers):\n{payload}"
