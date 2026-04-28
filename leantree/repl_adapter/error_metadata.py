from __future__ import annotations

from typing import Any


def first_error_line(text: str) -> str:
    normalized = text.strip().replace("\\r\\n", "\n").replace("\\n", "\n")
    return normalized.splitlines()[0].strip()


def normalize_repl_error_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not isinstance(messages, list):
        return None
    normalized = [entry for entry in messages if isinstance(entry, dict)]
    return normalized or None


def summarize_repl_error_messages(messages: list[dict[str, Any]]) -> str:
    for entry in messages:
        data = entry.get("data")
        if isinstance(data, str) and data.strip():
            return first_error_line(data)
    return "Lean REPL error"


def build_error_record(error: str | Exception) -> dict[str, Any]:
    raw = error if isinstance(error, str) else str(error)
    record: dict[str, Any] = {"error": raw}

    error_kind = getattr(error, "error_kind", None)
    error_summary = getattr(error, "error_summary", None)
    repl_messages = normalize_repl_error_messages(getattr(error, "repl_messages", None))
    if repl_messages is not None:
        record["repl_messages"] = repl_messages
        if not isinstance(error_kind, str) or not error_kind:
            error_kind = "repl"
        if not isinstance(error_summary, str) or not error_summary:
            error_summary = summarize_repl_error_messages(repl_messages)
    if isinstance(error_kind, str) and error_kind:
        record["error_kind"] = error_kind
    if isinstance(error_summary, str) and error_summary:
        record["error_summary"] = error_summary
    elif isinstance(raw, str) and raw.strip():
        record["error_summary"] = first_error_line(raw)
    return record
