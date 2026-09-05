"""Build the Deep Agents coding assistant used by the Streamlit app."""

import os
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv

load_dotenv()

WORKSPACE_ROOT = Path(__file__).parent.joinpath("workspace").resolve()
SKILLS_ROOT = WORKSPACE_ROOT.joinpath("skills")

SYSTEM_PROMPT = """
You are a small educational coding agent.

Work only inside the provided virtual workspace. Inspect existing files before
editing them. For tasks with more than one step, write a short plan first. Make
the smallest useful change, preserve existing behavior, and explain which files
you changed.

Use the code-reviewer subagent after making a meaningful code change. The
reviewer should look for correctness problems, missing edge cases, and unclear
behavior. Apply useful reviewer feedback before giving the final answer.

When the user explicitly requests a named skill, read that skill's SKILL.md
before inspecting or editing project files and follow its workflow.

You cannot run shell commands in this exercise. Be transparent about that
limitation and tell the learner how to run or test the code locally.
""".strip()

REVIEWER_SUBAGENT = {
    "name": "code-reviewer",
    "description": (
        "Review code changes for correctness, edge cases, and clarity before "
        "the main agent finishes."
    ),
    "system_prompt": """
You are a concise code reviewer. Read the relevant workspace files, identify
only concrete correctness or maintainability problems, and return a short list
of fixes. Do not edit files yourself.
""".strip(),
}


def build_coding_agent() -> Any:
    """Create a Deep Agent whose file tools are restricted to the lab workspace."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    WORKSPACE_ROOT.mkdir(exist_ok=True)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    return create_deep_agent(
        model=f"google_genai:{model_name}",
        system_prompt=SYSTEM_PROMPT,
        subagents=[REVIEWER_SUBAGENT],
        skills=["skills/"],
        backend=FilesystemBackend(
            root_dir=WORKSPACE_ROOT,
            virtual_mode=True,
        ),
        name="learning-coding-agent",
    )
