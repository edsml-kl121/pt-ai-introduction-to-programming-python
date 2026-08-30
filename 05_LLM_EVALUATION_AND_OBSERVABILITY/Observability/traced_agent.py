"""Trace a Gemini tool-calling agent with LangChain and LangSmith."""

import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

WEATHER = {
    "bangkok": {
        "celsius": 32,
        "fahrenheit": 90,
        "condition": "partly cloudy",
    },
    "tokyo": {
        "celsius": 24,
        "fahrenheit": 75,
        "condition": "clear",
    },
}

WEATHER_ALERTS = {
    "bangkok": ["Heat advisory until 18:00 local time."],
    "tokyo": [],
}


@tool
def get_weather(
    city: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    """Get the current weather for a supported city."""
    weather = WEATHER.get(city.lower())
    if weather is None:
        return json.dumps(
            {
                "error": f"No weather data is available for {city}.",
                "supported_cities": sorted(WEATHER),
            }
        )

    return json.dumps(
        {
            "city": city,
            "temperature": weather[unit],
            "unit": unit,
            "condition": weather["condition"],
        }
    )


@tool
def get_weather_alerts(city: str) -> str:
    """Get active weather alerts for a supported city."""
    alerts = WEATHER_ALERTS.get(city.lower())
    if alerts is None:
        return json.dumps(
            {
                "error": f"No alert data is available for {city}.",
                "supported_cities": sorted(WEATHER_ALERTS),
            }
        )
    return json.dumps({"city": city, "alerts": alerts})


model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[get_weather, get_weather_alerts],
    system_prompt=(
        "You are a weather assistant. Always call the available tools for current "
        "weather or alert questions. Do not invent weather data. Clearly summarize "
        "the tool results and mention when there are no active alerts."
    ),
)


def message_text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    text = "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if not text:
        raise ValueError("The agent response did not contain a text block.")
    return text


def run_agent(question: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "run_name": "observed_weather_agent",
            "tags": ["workshop", "langchain", "agent", "tool-calling"],
            "metadata": {
                "lesson": "05",
                "agent_name": "weather_agent",
                "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            },
        },
    )
    return message_text(result["messages"][-1].content)


def main() -> None:
    question = (
        "Compare the current weather in Bangkok and Tokyo in Celsius. "
        "Tell me which city is warmer and check whether either city has an alert."
    )
    print(f"Question: {question}\n")
    print(f"Agent: {run_agent(question)}")


if __name__ == "__main__":
    main()
