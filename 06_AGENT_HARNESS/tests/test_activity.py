"""Tests for safe Deep Agents activity rendering."""

import unittest

from langchain_core.messages import AIMessage, ToolMessage

from activity import activity_events, final_answer, render_activity


class ActivityTests(unittest.TestCase):
    def test_extracts_skill_discovery(self) -> None:
        events = activity_events(
            {
                "type": "updates",
                "ns": (),
                "data": {
                    "SkillsMiddleware.before_agent": {
                        "skills_metadata": [
                            {"name": "python-testing"},
                            {"name": "python-refactor"},
                        ]
                    }
                },
            }
        )

        self.assertEqual(events[0]["title"], "Skills discovered")
        self.assertIn("python-testing", events[0]["detail"])

    def test_extracts_tool_call_and_result(self) -> None:
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "/calculator.py"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        result = ToolMessage(
            content="file contents",
            name="read_file",
            tool_call_id="call-1",
        )

        call_events = activity_events(
            {
                "type": "updates",
                "ns": (),
                "data": {"model": {"messages": [call]}},
            }
        )
        result_events = activity_events(
            {
                "type": "updates",
                "ns": (),
                "data": {"tools": {"messages": [result]}},
            }
        )

        self.assertEqual(call_events[0]["title"], "Main agent called `read_file`")
        self.assertIn("/calculator.py", call_events[0]["detail"])
        self.assertEqual(result_events[0]["title"], "`read_file` returned")

    def test_labels_subagent_events(self) -> None:
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "grep",
                    "args": {"pattern": "add"},
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        )
        events = activity_events(
            {
                "type": "updates",
                "ns": ("tools:subagent-1",),
                "data": {"model": {"messages": [message]}},
            }
        )

        self.assertEqual(events[0]["title"], "Subagent called `grep`")

    def test_extracts_only_top_level_final_answer(self) -> None:
        part = {
            "type": "updates",
            "ns": (),
            "data": {"model": {"messages": [AIMessage(content="Done")]}},
        }

        self.assertEqual(final_answer(part), "Done")
        self.assertIsNone(final_answer({**part, "ns": ("tools:subagent-1",)}))

    def test_renders_numbered_activity(self) -> None:
        markdown = render_activity(
            [{"title": "Main agent called `ls`", "detail": '{"path": "/"}'}]
        )

        self.assertIn("**1. Main agent called `ls`**", markdown)
        self.assertIn('{"path": "/"}', markdown)


if __name__ == "__main__":
    unittest.main()
