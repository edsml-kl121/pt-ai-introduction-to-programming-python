"""Tests for Streamlit session-message normalization."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from message_utils import normalize_message


class MessageNormalizationTests(unittest.TestCase):
    def test_normalizes_legacy_human_message(self) -> None:
        message = normalize_message(HumanMessage(content="Hello"))

        self.assertEqual(message, {"role": "user", "content": "Hello"})

    def test_normalizes_legacy_ai_message(self) -> None:
        message = normalize_message(AIMessage(content="Hi"))

        self.assertEqual(message, {"role": "assistant", "content": "Hi"})

    def test_preserves_slash_command_display_text(self) -> None:
        message = normalize_message(
            {
                "role": "user",
                "content": "Expanded agent instruction",
                "display": "/python-testing add tests",
            }
        )

        self.assertEqual(message["display"], "/python-testing add tests")


if __name__ == "__main__":
    unittest.main()
