"""Streamlit interface for the Deep Agents coding assistant."""

from pathlib import Path
from typing import Any

import streamlit as st
from st_keyup import st_keyup

from activity import activity_events, final_answer, render_activity
from agent import SKILLS_ROOT, WORKSPACE_ROOT, build_coding_agent
from commands import (
    discover_skills,
    route_prompt,
    skill_list,
    slash_suggestions,
)
from message_utils import normalize_message

st.set_page_config(page_title="Learning Coding Agent", page_icon="🧰")


def workspace_files() -> list[Path]:
    """List regular workspace files while excluding generated Python caches."""
    return sorted(
        path
        for path in WORKSPACE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def agent_messages() -> list[dict[str, str]]:
    """Remove display-only fields before sending conversation state to the agent."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]


def reset_composer() -> None:
    """Clear the keyup component by mounting it with a new Streamlit key."""
    st.session_state.composer_default = ""
    st.session_state.composer_generation += 1


@st.cache_resource
def coding_agent() -> Any:
    return build_coding_agent()


st.title("Learning Coding Agent")
st.caption(
    "Deep Agents + LangChain + Gemini with workspace skills and slash commands"
)
st.info(
    "Open **Agent activity** under a response to inspect plans, tool calls, "
    "skill reads, results, and subagent steps. Private hidden chain-of-thought "
    "is not exposed."
)

skills = discover_skills(SKILLS_ROOT)

with st.sidebar:
    st.header("Skills")
    st.markdown(skill_list(skills))
    st.caption("Try `/python-testing add tests for calculator.py`")

    st.header("Workspace")
    files = workspace_files()
    if files:
        selected = st.selectbox(
            "Inspect a file",
            files,
            format_func=lambda path: str(path.relative_to(WORKSPACE_ROOT)),
        )
        language = "markdown" if selected.suffix == ".md" else "python"
        st.code(selected.read_text(encoding="utf-8"), language=language)
    else:
        st.info("The workspace is empty.")

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.messages = [
    normalize_message(message)
    for message in st.session_state.messages
]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        activity = message.get("activity")
        if activity:
            with st.expander("Agent activity", expanded=False):
                st.markdown(render_activity(activity))
        st.markdown(message.get("display", message["content"]))

st.subheader("Prompt")
if "composer_generation" not in st.session_state:
    st.session_state.composer_generation = 0

composer_default = st.session_state.pop("composer_default", "")
composer = st_keyup(
    "Message",
    value=composer_default,
    debounce=100,
    key=f"prompt-composer-{st.session_state.composer_generation}",
)
suggestions = slash_suggestions(composer or "", skills)

if (composer or "").startswith("/"):
    with st.container(border=True):
        st.caption("Slash commands")
        if suggestions:
            for skill in suggestions:
                command_col, description_col = st.columns([1, 3])
                with command_col:
                    if st.button(
                        f"/{skill.name}",
                        key=f"select-{skill.name}",
                        use_container_width=True,
                    ):
                        st.session_state.composer_default = f"/{skill.name} "
                        st.session_state.composer_generation += 1
                        st.rerun()
                with description_col:
                    st.write(skill.description)
        else:
            st.caption("No matching skills.")

        if st.button("Show all skills", use_container_width=False):
            st.session_state.pending_prompt = "/skills"
            st.rerun()

palette_prompt = st.session_state.pop("pending_prompt", None)
send_prompt = st.button(
    "Send",
    type="primary",
    disabled=not bool((composer or "").strip()),
)
prompt = palette_prompt or ((composer or "").strip() if send_prompt else None)
if prompt:
    routed_prompt, local_response = route_prompt(prompt, skills)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": routed_prompt or prompt,
            "display": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    if local_response:
        with st.chat_message("assistant"):
            st.markdown(local_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": local_response}
        )
        reset_composer()
        st.rerun()
    else:
        assert routed_prompt is not None
        try:
            with st.chat_message("assistant"):
                events: list[dict[str, str]] = []
                answer = ""
                with st.expander("Agent activity", expanded=True):
                    activity_panel = st.empty()
                    activity_panel.markdown(render_activity(events))

                    for part in coding_agent().stream(
                        {"messages": agent_messages()},
                        stream_mode="updates",
                        subgraphs=True,
                        version="v2",
                    ):
                        events.extend(activity_events(part))
                        streamed_answer = final_answer(part)
                        if streamed_answer:
                            answer = streamed_answer
                        activity_panel.markdown(render_activity(events))

                if not answer:
                    raise RuntimeError("The agent completed without a final response.")
                st.markdown(answer)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "activity": events,
                }
            )
            reset_composer()
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
