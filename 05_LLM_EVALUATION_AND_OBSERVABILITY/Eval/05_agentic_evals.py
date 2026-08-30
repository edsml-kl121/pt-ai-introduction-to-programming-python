"""Evaluate agent trajectories with AgentEvals and Gemini."""

import json

from agentevals.trajectory.llm import create_trajectory_llm_as_judge
from agentevals.trajectory.match import create_trajectory_match_evaluator
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openevals.prompts import TOOL_SELECTION_PROMPT

from model import get_chat_model

load_dotenv()

INTENT_RESOLUTION_PROMPT = """
Evaluate whether the actual agent trajectory correctly understood and resolved
the user's intent. Consider the user's request, tool calls, tool results, and
final response. Compare it with the reference trajectory when one is supplied.

<reference_trajectory>{reference_outputs}</reference_trajectory>
<actual_trajectory>{outputs}</actual_trajectory>
"""

TASK_ADHERENCE_PROMPT = """
Evaluate whether the actual agent trajectory followed the task rules represented
by the reference trajectory. Penalize wrong or unnecessary tools, invented tool
results, skipped required steps, and final claims unsupported by tool results.

<reference_trajectory>{reference_outputs}</reference_trajectory>
<actual_trajectory>{outputs}</actual_trajectory>
"""

judge = get_chat_model()

strict_match = create_trajectory_match_evaluator(
    trajectory_match_mode="strict",
    tool_args_match_mode="exact",
)
tool_selection_judge = create_trajectory_llm_as_judge(
    prompt=TOOL_SELECTION_PROMPT,
    judge=judge,
    feedback_key="tool_selection",
    continuous=True,
)
trajectory_quality_judge = create_trajectory_llm_as_judge(
    judge=judge,
    feedback_key="trajectory_accuracy",
    continuous=True,
)
intent_resolution_judge = create_trajectory_llm_as_judge(
    prompt=INTENT_RESOLUTION_PROMPT,
    judge=judge,
    feedback_key="intent_resolution",
    continuous=True,
)
task_adherence_judge = create_trajectory_llm_as_judge(
    prompt=TASK_ADHERENCE_PROMPT,
    judge=judge,
    feedback_key="task_adherence",
    continuous=True,
)


def evaluate_agent_run(
    trajectory: list,
    reference_trajectory: list,
) -> dict:
    return {
        "strict_trajectory_match": strict_match(
            outputs=trajectory,
            reference_outputs=reference_trajectory,
        ),
        "tool_selection": tool_selection_judge(
            outputs=trajectory,
            reference_outputs=reference_trajectory,
        ),
        "trajectory_accuracy": trajectory_quality_judge(
            outputs=trajectory,
            reference_outputs=reference_trajectory,
        ),
        "intent_resolution": intent_resolution_judge(
            outputs=trajectory,
            reference_outputs=reference_trajectory,
        ),
        "task_adherence": task_adherence_judge(
            outputs=trajectory,
            reference_outputs=reference_trajectory,
        ),
    }


def main() -> None:
    reference_trajectory = [
        HumanMessage(content="What is the weather in Tokyo in Celsius?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "weather-call",
                    "name": "get_weather",
                    "args": {"city": "Tokyo", "unit": "celsius"},
                }
            ],
        ),
        ToolMessage(
            content='{"temperature": 22, "condition": "clear"}',
            tool_call_id="weather-call",
        ),
        AIMessage(content="Tokyo is currently 22 C with clear skies."),
    ]

    correct_trajectory = list(reference_trajectory)
    incorrect_trajectory = [
        HumanMessage(content="What is the weather in Tokyo in Celsius?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "flight-call",
                    "name": "book_flight",
                    "args": {
                        "origin": "London",
                        "destination": "Paris",
                        "date": "2026-09-01",
                    },
                }
            ],
        ),
        ToolMessage(
            content='{"confirmation": "ABC123"}',
            tool_call_id="flight-call",
        ),
        AIMessage(content="Tokyo is currently 22 C with clear skies."),
    ]

    samples = [
        ("Correct weather trajectory", correct_trajectory),
        ("Wrong tool and unsupported claim", incorrect_trajectory),
    ]

    for name, trajectory in samples:
        print(f"\n=== {name} ===")
        result = evaluate_agent_run(trajectory, reference_trajectory)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
