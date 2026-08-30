"""Create the LangSmith dataset used by the evaluation lab."""

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "mew-tutor-langchain-evaluation"

EXAMPLES = [
    {
        "inputs": {
            "question": "What is photosynthesis?",
            "context": (
                "Photosynthesis occurs in plant-cell chloroplasts. Plants use "
                "sunlight, water, and carbon dioxide to produce glucose and oxygen."
            ),
        },
        "outputs": {
            "answer": (
                "Photosynthesis is the process plants use to convert light energy "
                "into chemical energy stored as glucose."
            )
        },
    },
    {
        "inputs": {
            "question": "What causes rain?",
            "context": (
                "Warm air rises and cools, causing water vapor to condense into "
                "cloud droplets. Droplets fall when they become heavy enough."
            ),
        },
        "outputs": {
            "answer": (
                "Rain occurs when condensed water droplets in clouds become heavy "
                "enough to fall to the ground."
            )
        },
    },
    {
        "inputs": {
            "question": "How does a battery work?",
            "context": (
                "A battery contains electrochemical cells with an anode, cathode, "
                "and electrolyte. Chemical reactions cause electrons to flow "
                "through an external circuit."
            ),
        },
        "outputs": {
            "answer": (
                "A battery converts chemical energy into electrical energy through "
                "electrochemical reactions that drive electrons through a circuit."
            )
        },
    },
]


def main() -> None:
    client = Client()
    matches = list(client.list_datasets(dataset_name=DATASET_NAME, limit=1))

    if matches:
        print(f'Dataset "{DATASET_NAME}" already exists with ID {matches[0].id}.')
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Question-answer examples for the LangChain evaluation lab.",
    )
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    print(f'Created dataset "{DATASET_NAME}" with ID {dataset.id}.')


if __name__ == "__main__":
    main()
