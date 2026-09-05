"""Skill discovery and slash-command routing for the Streamlit interface."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path


def discover_skills(skills_root: Path) -> dict[str, Skill]:
    """Read skill metadata from each SKILL.md under the supplied directory."""
    skills: dict[str, Skill] = {}
    if not skills_root.exists():
        return skills

    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            raise ValueError(f"Invalid frontmatter in {skill_file}")

        metadata = yaml.safe_load(parts[1])
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata in {skill_file}")

        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            raise ValueError(f"Skill name and description are required in {skill_file}")

        skills[name] = Skill(
            name=name,
            description=description.strip(),
            path=skill_file,
        )

    return skills


def skill_list(skills: dict[str, Skill]) -> str:
    """Format the available slash commands for display."""
    if not skills:
        return "No skills are installed."
    lines = ["Available skills:"]
    lines.extend(
        f"- `/{skill.name} <request>`: {skill.description}"
        for skill in skills.values()
    )
    return "\n".join(lines)


def slash_suggestions(
    query: str,
    skills: dict[str, Skill],
) -> list[Skill]:
    """Return skills whose slash command matches the current input."""
    stripped = query.strip()
    if not stripped.startswith("/"):
        return []

    command = stripped.partition(" ")[0].removeprefix("/").lower()
    return [
        skill
        for skill in skills.values()
        if skill.name.lower().startswith(command)
    ]


def route_prompt(
    prompt: str,
    skills: dict[str, Skill],
) -> tuple[str | None, str | None]:
    """Return an agent prompt or a local response for a slash command."""
    stripped = prompt.strip()
    if not stripped.startswith("/"):
        return stripped, None

    command, _, request = stripped.partition(" ")
    name = command.removeprefix("/")

    if name in {"help", "skills"}:
        return None, skill_list(skills)

    skill = skills.get(name)
    if skill is None:
        return None, (
            f"Unknown skill `/{name}`.\n\n"
            f"{skill_list(skills)}"
        )
    if not request.strip():
        return None, f"Usage: `/{name} <request>`"

    agent_prompt = (
        f"Explicitly activate the `{name}` skill. Read its SKILL.md before "
        f"taking any other action, then follow its instructions.\n\n"
        f"Task: {request.strip()}"
    )
    return agent_prompt, None
