"""Shared helpers for working with OpenAI's Python SDK."""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------

def get_openai_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Return an OpenAI client or ``None`` if no API key is configured."""
    if not api_key:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Chat completion helper
# ---------------------------------------------------------------------------

def call_chat_completion(
    client: Optional[OpenAI],
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1500,
    temperature: float = 0.7,
    system_message: str = "You are a helpful assistant that provides comprehensive answers based on document content.",
) -> str:
    """Send a single-prompt chat completion request and return the text.

    A lightweight wrapper that gracefully handles the case where ``client``
    is ``None`` and captures exceptions so callers can surface the error
    message while keeping the original logging semantics.
    """
    if client is None:
        return "OpenAI client not available. Please check your API key configuration."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as exc:  # Broad except to propagate the error string
        return f"Error generating LLM response: {exc}" 