"""Convert Deep Agents stream updates into safe, readable UI activity."""

import json
from typing import Any

from message_utils import message_text

MAX_DETAIL_LENGTH = 4000


def _source_label(namespace: tuple[str, ...]) -> str:
    if not namespace:
        return "Main agent"
    if any(segment.startswith("tools:") for segment in namespace):
        return "Subagent"
    return "Agent component"


def _detail(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, indent=2, ensure_ascii=True, default=str)
        except TypeError:
            text = str(value)
    if len(text) > MAX_DETAIL_LENGTH:
        return f"{text[:MAX_DETAIL_LENGTH]}\n... output truncated ..."
    return text


def activity_events(part: dict[str, Any]) -> list[dict[str, str]]:
    """Extract observable actions from one LangGraph v2 update."""
    if part.get("type") != "updates":
        return []

    namespace = tuple(part.get("ns", ()))
    source = _source_label(namespace)
    events: list[dict[str, str]] = []

    for node_name, update in part.get("data", {}).items():
        if node_name == "SkillsMiddleware.before_agent" and isinstance(update, dict):
            names = [
                metadata["name"]
                for metadata in update.get("skills_metadata", [])
                if isinstance(metadata, dict) and isinstance(metadata.get("name"), str)
            ]
            if names:
                events.append(
                    {
                        "title": "Skills discovered",
                        "detail": ", ".join(f"`{name}`" for name in names),
                    }
                )
            continue

        if not isinstance(update, dict):
            continue

        for message in update.get("messages", []):
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                name = tool_call.get("name", "unknown")
                events.append(
                    {
                        "title": f"{source} called `{name}`",
                        "detail": _detail(tool_call.get("args", {})),
                    }
                )

            message_type = getattr(message, "type", "")
            if message_type == "tool":
                name = getattr(message, "name", None) or "tool"
                events.append(
                    {
                        "title": f"`{name}` returned",
                        "detail": _detail(message_text(message)),
                    }
                )
                continue

            text = message_text(message).strip()
            if text and not tool_calls:
                usage = getattr(message, "usage_metadata", None) or {}
                reasoning_tokens = (
                    usage.get("output_token_details", {}).get("reasoning")
                    if isinstance(usage, dict)
                    else None
                )
                title = f"{source} produced a response"
                if isinstance(reasoning_tokens, int):
                    title += f" ({reasoning_tokens} reasoning tokens)"
                events.append({"title": title, "detail": _detail(text)})

    return events


def final_answer(part: dict[str, Any]) -> str | None:
    """Return a completed top-level model answer from a stream update."""
    if part.get("type") != "updates" or part.get("ns"):
        return None

    model_update = part.get("data", {}).get("model")
    if not isinstance(model_update, dict):
        return None

    for message in reversed(model_update.get("messages", [])):
        if getattr(message, "type", "") != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = message_text(message).strip()
        if text:
            return text
    return None


def render_activity(events: list[dict[str, str]]) -> str:
    """Format accumulated events as Markdown for Streamlit."""
    if not events:
        return "_Waiting for the first agent event..._"

    sections = []
    for index, event in enumerate(events, start=1):
        detail = event["detail"]
        sections.append(
            f"**{index}. {event['title']}**\n\n"
            f"```text\n{detail}\n```"
        )
    return "\n\n".join(sections)
