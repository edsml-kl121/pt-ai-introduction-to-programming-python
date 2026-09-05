"""Helpers for keeping Streamlit and LangChain message formats separate."""

from typing import Any


def message_text(message: Any) -> str:
    """Return readable text from a LangChain message or message content value."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_message(message: Any) -> dict[str, Any]:
    """Convert legacy LangChain messages into the app's dictionary format."""
    if isinstance(message, dict):
        normalized = {
            "role": str(message.get("role", "assistant")),
            "content": message_text(message.get("content", "")),
        }
        display = message.get("display")
        if isinstance(display, str):
            normalized["display"] = display
        activity = message.get("activity")
        if isinstance(activity, list):
            normalized["activity"] = activity
        return normalized

    message_type = getattr(message, "type", "ai")
    role = {
        "human": "user",
        "ai": "assistant",
        "system": "assistant",
        "tool": "assistant",
    }.get(message_type, "assistant")
    return {
        "role": role,
        "content": message_text(message),
    }
