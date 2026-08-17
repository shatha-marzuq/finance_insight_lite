"""Small, shared Groq client helpers used by the RAG and evaluation layers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_groq import ChatGroq


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
_TRACING_ENV_VARS = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")


@contextmanager
def _without_langchain_tracing() -> Any:
    original = {key: os.environ.get(key) for key in _TRACING_ENV_VARS}
    try:
        for key in _TRACING_ENV_VARS:
            os.environ[key] = "false"
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def create_groq_chat_model(
    *, model: str = DEFAULT_GROQ_MODEL, temperature: float = 0
) -> Any:
    """Create the project's deterministic Groq chat model.

    The API key is read only when a caller actually creates a model.  This
    keeps offline evaluation (and its tests) usable without a Groq key.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required when --with-judge is enabled.")
    # Kept lazy so offline metric evaluation and mocked judge tests do not
    # require the optional LangChain/Groq runtime to be installed.
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, api_key=SecretStr(api_key),max_tokens=1024, temperature=temperature)


class GroqJudgeClient:
    """Adapter for a LangChain-compatible Groq model used by LLM judges."""

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, chat_model: Any | None = None):
        self.model = model
        self._chat_model = chat_model

    @property
    def chat_model(self) -> Any:
        if self._chat_model is None:
            with _without_langchain_tracing():
                self._chat_model = create_groq_chat_model(model=self.model, temperature=0)
        return self._chat_model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        with _without_langchain_tracing():
            response = self.chat_model.invoke(
                [("system", system_prompt), ("human", user_prompt)]
            )
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        return str(content)
