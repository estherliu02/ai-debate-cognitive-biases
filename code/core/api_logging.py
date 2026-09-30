from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


_SECRET_KEY_PARTS = ("authorization", "api_key", "apikey", "secret", "password", "token")


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in _SECRET_KEY_PARTS):
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _sanitize_for_log(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_log(item) for item in value]
    return value


def _format_json_block(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class DialogueApiLogger:
    """Append-only logger for dialogue-step API attempts."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        jsonl_filename: str = "dialogue_api_log.jsonl",
        text_filename: str = "dialogue_api_log.txt",
    ) -> None:
        self.output_root = Path(output_root)
        self.jsonl_path = self.output_root / jsonl_filename
        self.text_path = self.output_root / text_filename
        self._lock = threading.Lock()

    def log_attempt(
        self,
        *,
        context: dict[str, Any] | None,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        status: str,
        error: str | None,
    ) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        base_context = dict(context or {})
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": base_context.get("case_id"),
            "condition": base_context.get("condition"),
            "ground_truth_order": base_context.get("ground_truth_order"),
            "role": base_context.get("role"),
            "speaker": base_context.get("speaker"),
            "turn_id": base_context.get("turn_id"),
            "speech_type": base_context.get("speech_type"),
            "attempt": base_context.get("attempt"),
            "model": base_context.get("model"),
            "generation_parameters": _sanitize_for_log(base_context.get("generation_parameters") or {}),
            "request": _sanitize_for_log(request),
            "response": _sanitize_for_log(response or {"raw_text": None, "parsed_output": None}),
            "status": status,
            "error": error,
        }
        text_record = self._format_text_record(record)
        with self._lock:
            self._append_jsonl(record)
            self._append_text(text_record)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append_text(self, text: str) -> None:
        with self.text_path.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _format_text_record(record: dict[str, Any]) -> str:
        request = record.get("request") or {}
        response = record.get("response") or {}
        messages = request.get("messages") or []
        message_text = _format_json_block(messages)
        return (
            "================================================================================\n"
            f"TIMESTAMP: {record.get('timestamp')}\n"
            f"CASE: {record.get('case_id')}\n"
            f"CONDITION: {record.get('condition')}\n"
            f"GROUND TRUTH ORDER: {record.get('ground_truth_order')}\n"
            f"TURN: {record.get('turn_id')} ({record.get('speech_type')})\n"
            f"ROLE: {record.get('role')}\n"
            f"SPEAKER: {record.get('speaker')}\n"
            f"ATTEMPT: {record.get('attempt')}\n"
            f"MODEL: {record.get('model')}\n\n"
            "[SYSTEM PROMPT]\n"
            f"{request.get('system_prompt') or ''}\n\n"
            "[USER PROMPT]\n"
            f"{request.get('user_prompt') or ''}\n\n"
            "[MESSAGES]\n"
            f"{message_text}\n\n"
            "[GENERATION PARAMETERS]\n"
            f"{_format_json_block(record.get('generation_parameters'))}\n\n"
            "[RAW RESPONSE]\n"
            f"{response.get('raw_text') or ''}\n\n"
            "[PARSED RESPONSE]\n"
            f"{_format_json_block(response.get('parsed_output'))}\n\n"
            "[ERROR]\n"
            f"{record.get('error') or ''}\n"
        )
