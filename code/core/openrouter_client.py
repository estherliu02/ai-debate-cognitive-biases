from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import requests


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


_RETRY_ATTEMPTS = _env_int("OPENROUTER_RETRY_ATTEMPTS", 6)
_RETRY_DELAY = _env_int("OPENROUTER_RETRY_DELAY", 5, minimum=0)
_RETRY_MAX_DELAY = _env_int("OPENROUTER_RETRY_MAX_DELAY", 60, minimum=1)
_REQUEST_TIMEOUT = _env_int("OPENROUTER_TIMEOUT", 180)
_ERROR_SNIPPET_LEN = 400
_MAX_JSON_RETRY_TOKENS = 6000
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_RETRYABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)


class OpenRouterResponseError(RuntimeError):
    """Raised when OpenRouter returns an unexpected response payload."""


def _retry_delay_for_attempt(attempt: int) -> int:
    return min(_RETRY_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)


def _retry_after_seconds(resp: requests.Response) -> int | None:
    retry_after = resp.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return max(0, int(retry_after))
    except ValueError:
        return None


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY. Set it in your environment.")
        self.base_url = base_url.rstrip("/")
        self.last_json_request_payload: dict[str, Any] | None = None
        self.last_json_response_raw_text: str | None = None
        self.last_json_parsed_output: dict[str, Any] | None = None
        self.last_json_finish_reason: str | None = None
        self.last_json_attempt_records: list[dict[str, Any]] = []

    def _post(self, payload: dict) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT)
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    last_exc = requests.exceptions.HTTPError(
                        f"OpenRouter returned retryable HTTP status {resp.status_code}",
                        response=resp,
                    )
                    delay = _retry_after_seconds(resp) or _retry_delay_for_attempt(attempt)
                    if attempt < _RETRY_ATTEMPTS:
                        print(
                            f"[openrouter] attempt {attempt}/{_RETRY_ATTEMPTS} failed "
                            f"(HTTP {resp.status_code}). Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        continue
                    break
                resp.raise_for_status()
                return resp
            except _RETRYABLE_REQUEST_EXCEPTIONS as e:
                last_exc = e
                delay = _retry_delay_for_attempt(attempt)
                if attempt < _RETRY_ATTEMPTS:
                    print(
                        f"[openrouter] attempt {attempt}/{_RETRY_ATTEMPTS} failed "
                        f"({type(e).__name__}). Retrying in {delay}s..."
                    )
                    time.sleep(delay)
        raise last_exc

    def _response_snippet(self, text: str) -> str:
        compact = " ".join(text.split())
        return compact[:_ERROR_SNIPPET_LEN]

    def _decode_response_json(self, resp: requests.Response) -> dict:
        try:
            return resp.json()
        except requests.exceptions.JSONDecodeError as exc:
            snippet = self._response_snippet(resp.text)
            raise OpenRouterResponseError(
                f"OpenRouter returned a non-JSON HTTP body "
                f"(status={resp.status_code}): {snippet}"
            ) from exc

    def _coerce_content_to_text(self, content: Any) -> str | None:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts) if parts else None
        return None

    def _extract_message_content(self, body: dict) -> str:
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            snippet = self._response_snippet(json.dumps(body, ensure_ascii=True))
            raise OpenRouterResponseError(
                f"OpenRouter response is missing choices/message/content: {snippet}"
            ) from exc
        content = self._coerce_content_to_text(message.get("content"))
        if content is None or not content.strip():
            snippet = self._response_snippet(json.dumps(message, ensure_ascii=True))
            raise OpenRouterResponseError(
                f"OpenRouter response message content is empty or non-text: {snippet}"
            )
        return content

    def _extract_finish_reason(self, body: dict) -> str | None:
        try:
            return body["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            return None

    def _extract_json_candidate(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return stripped[start : end + 1]
        return stripped

    def _looks_truncated_json(self, content: str) -> bool:
        candidate = self._extract_json_candidate(content)
        return candidate.startswith("{") and not candidate.endswith("}")

    def complete_json(self, model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 1200) -> Dict[str, Any]:
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "You are a careful assistant that always returns valid JSON when asked."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        self.last_json_request_payload = dict(payload)
        self.last_json_response_raw_text = None
        self.last_json_parsed_output = None
        self.last_json_finish_reason = None
        self.last_json_attempt_records = []
        last_exc = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            content = ""
            finish_reason = None
            request_payload = dict(payload)
            try:
                self.last_json_request_payload = dict(request_payload)
                resp = self._post(payload)
                body = self._decode_response_json(resp)
                content = self._extract_message_content(body)
                finish_reason = self._extract_finish_reason(body)
                self.last_json_response_raw_text = content
                self.last_json_finish_reason = finish_reason
                parsed = json.loads(self._extract_json_candidate(content))
                self.last_json_parsed_output = parsed
                self.last_json_attempt_records.append({
                    "request_payload": request_payload,
                    "raw_text": content,
                    "parsed_output": parsed,
                    "finish_reason": finish_reason,
                    "status": "success",
                    "error": None,
                })
                return parsed
            except (json.JSONDecodeError, OpenRouterResponseError) as exc:
                last_exc = exc
                self.last_json_response_raw_text = content or self.last_json_response_raw_text
                self.last_json_finish_reason = finish_reason
                self.last_json_attempt_records.append({
                    "request_payload": request_payload,
                    "raw_text": content or None,
                    "parsed_output": None,
                    "finish_reason": finish_reason,
                    "status": "error",
                    "error": str(exc),
                })
                snippet = ""
                if isinstance(exc, json.JSONDecodeError):
                    snippet = self._response_snippet(content)
                    should_expand_budget = (
                        finish_reason == "length" or self._looks_truncated_json(content)
                    )
                    if should_expand_budget and payload["max_tokens"] < _MAX_JSON_RETRY_TOKENS:
                        payload["max_tokens"] = min(
                            int(payload["max_tokens"] * 1.5),
                            _MAX_JSON_RETRY_TOKENS,
                        )
                    print(
                        f"[openrouter] attempt {attempt}/{_RETRY_ATTEMPTS} returned malformed JSON content. "
                        f"Retrying in {_RETRY_DELAY}s... finish_reason={finish_reason!r} "
                        f"max_tokens={payload['max_tokens']} snippet={snippet}"
                    )
                else:
                    if payload["max_tokens"] < _MAX_JSON_RETRY_TOKENS:
                        payload["max_tokens"] = min(
                            int(payload["max_tokens"] * 2),
                            _MAX_JSON_RETRY_TOKENS,
                        )
                    print(
                        f"[openrouter] attempt {attempt}/{_RETRY_ATTEMPTS} returned an invalid response. "
                        f"Retrying in {_RETRY_DELAY}s... max_tokens={payload['max_tokens']} {exc}"
                    )
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(_RETRY_DELAY)
        raise last_exc

    def complete_text(self, model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "You are a careful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        resp = self._post(payload)
        body = self._decode_response_json(resp)
        return self._extract_message_content(body)
