"""Shared HTTP plumbing for chat-completion providers."""

import requests

from rag_assistant.exceptions import LLMError


def post_json(
    session: requests.Session,
    url: str,
    payload: dict,
    timeout: float,
    provider: str,
    headers: dict[str, str] | None = None,
) -> dict:
    """POST `payload` and return the decoded JSON body, raising LLMError on any failure."""
    try:
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.Timeout as exc:
        raise LLMError(f"{provider} did not answer within {timeout:.0f}s") from exc
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach {provider} at {url}: {exc}") from exc

    if response.status_code != 200:
        # Bodies can be long; the first line is where providers put the reason.
        detail = response.text.strip().splitlines()[0][:300] if response.text.strip() else ""
        raise LLMError(f"{provider} returned HTTP {response.status_code}: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise LLMError(f"{provider} returned a response that is not JSON") from exc


def require_text(content: object, provider: str) -> str:
    """Validate that the provider produced non-empty text."""
    if not isinstance(content, str) or not content.strip():
        raise LLMError(f"{provider} returned an empty answer")
    return content.strip()
