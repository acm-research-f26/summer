"""
llm_client.py
-------------
Provider-agnostic wrapper around the Planner Agent's LLM call. Supports
Anthropic, OpenAI, and Google Gemini -- pick any one via `provider=`.
Each provider's SDK is imported lazily so you only need to `pip install`
the SDK(s) you actually use.

API keys are read from the standard env vars:
  ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY)
"""

from __future__ import annotations
import os


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str | None = None,
                 max_tokens: int = 2000, temperature: float = 0.0):
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key

        if self.provider not in {"anthropic", "openai", "gemini"}:
            raise ValueError(f"Unsupported provider '{provider}'. Use anthropic|openai|gemini.")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._complete_anthropic(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._complete_openai(system_prompt, user_prompt)
        elif self.provider == "gemini":
            return self._complete_gemini(system_prompt, user_prompt)
        raise RuntimeError("unreachable")

    # -- Anthropic -----------------------------------------------------
    def _complete_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

    # -- OpenAI ----------------------------------------------------------
    def _complete_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    # -- Gemini ------------------------------------------------------------
    def _complete_gemini(self, system_prompt: str, user_prompt: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key or os.environ.get("GOOGLE_API_KEY")
                         or os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
        resp = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature, max_output_tokens=self.max_tokens,
            ),
        )
        return resp.text or ""
