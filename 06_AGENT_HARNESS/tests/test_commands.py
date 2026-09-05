"""Tests for skill discovery and slash-command routing."""

import unittest
from pathlib import Path

from commands import discover_skills, route_prompt, slash_suggestions

SKILLS_ROOT = Path(__file__).parents[1] / "workspace" / "skills"


class SkillCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skills = discover_skills(SKILLS_ROOT)

    def test_discovers_both_sample_skills(self) -> None:
        self.assertEqual(
            set(self.skills),
            {"python-refactor", "python-testing"},
        )

    def test_routes_named_skill_to_explicit_agent_instruction(self) -> None:
        agent_prompt, local_response = route_prompt(
            "/python-testing add tests for calculator.py",
            self.skills,
        )

        self.assertIsNone(local_response)
        self.assertIn("Explicitly activate the `python-testing` skill", agent_prompt)
        self.assertIn("add tests for calculator.py", agent_prompt)

    def test_lists_skills_without_calling_agent(self) -> None:
        agent_prompt, local_response = route_prompt("/skills", self.skills)

        self.assertIsNone(agent_prompt)
        self.assertIn("/python-testing", local_response)
        self.assertIn("/python-refactor", local_response)

    def test_unknown_skill_returns_help(self) -> None:
        agent_prompt, local_response = route_prompt("/missing do work", self.skills)

        self.assertIsNone(agent_prompt)
        self.assertIn("Unknown skill `/missing`", local_response)

    def test_slash_opens_all_skill_suggestions(self) -> None:
        suggestions = slash_suggestions("/", self.skills)

        self.assertEqual(
            [skill.name for skill in suggestions],
            ["python-refactor", "python-testing"],
        )

    def test_slash_filters_skill_suggestions(self) -> None:
        suggestions = slash_suggestions("/python-t", self.skills)

        self.assertEqual(
            [skill.name for skill in suggestions],
            ["python-testing"],
        )

    def test_full_command_with_task_keeps_selected_skill_visible(self) -> None:
        suggestions = slash_suggestions(
            "/python-testing add tests",
            self.skills,
        )

        self.assertEqual(
            [skill.name for skill in suggestions],
            ["python-testing"],
        )

    def test_normal_prompt_does_not_show_suggestions(self) -> None:
        suggestions = slash_suggestions("please add tests", self.skills)

        self.assertEqual(suggestions, [])


if __name__ == "__main__":
    unittest.main()
