"""Trace a LangChain question-answering pipeline with LangSmith."""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

load_dotenv()

KNOWLEDGE_BASE = {
    "photosynthesis": (
        "Plants use sunlight, water, and carbon dioxide to make glucose and oxygen "
        "during photosynthesis."
    ),
    "rain": (
        "Rain forms when atmospheric water vapor condenses into droplets that "
        "become heavy enough to fall."
    ),
    "battery": (
        "Batteries use electrochemical reactions to convert chemical energy into "
        "electrical energy."
    ),
}


@traceable(name="retrieve_context", run_type="retriever")
def retrieve_context(question: str) -> str:
    """Return simple local context while creating a nested LangSmith run."""
    lowered_question = question.lower()
    matches = [
        text for keyword, text in KNOWLEDGE_BASE.items() if keyword in lowered_question
    ]
    return "\n".join(matches) or "No relevant context was found."


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the supplied context. Say when the context is insufficient.",
        ),
        ("human", "Question: {question}\n\nContext: {context}"),
    ]
)
chain = (
    prompt
    | ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
    )
    | StrOutputParser()
)


@traceable(name="observed_qa_application")
def answer_question(question: str) -> str:
    context = retrieve_context(question)
    return chain.invoke(
        {"question": question, "context": context},
        config={
            "run_name": "grounded_answer_chain",
            "tags": ["workshop", "langchain", "observability"],
            "metadata": {
                "lesson": "05",
                "context_source": "local_knowledge_base",
            },
        },
    )


def main() -> None:
    questions = [
        "What is photosynthesis?",
        "What causes rain?",
        "How does a battery work?",
        "What is the capital of Thailand?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer_question(question)}")


if __name__ == "__main__":
    main()
