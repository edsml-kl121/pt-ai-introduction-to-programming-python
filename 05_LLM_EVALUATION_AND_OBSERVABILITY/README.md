# LLM Evaluation and Observability

This session uses only the LangChain ecosystem and LangSmith for evaluating and observing an LLM application.

## Labs

1. [`Eval`](./Eval): create a LangSmith dataset, run a LangChain application against it, and compare deterministic and LLM-as-judge evaluators.
2. [`Observability`](./Observability): trace a LangChain pipeline in LangSmith and inspect its inputs, outputs, latency, tags, metadata, and nested runs.

The lab structure is adapted from these Azure workshop examples, with the Azure AI Evaluation SDK, Microsoft Agent Framework, Azure Monitor, and OpenTelemetry integrations replaced by LangChain and LangSmith:

- [Azure evaluation example](https://github.com/edsml-kl121/TH-azure-ai-workshop-2027/tree/main/LAB_5_evaluation)
- [Azure observability example](https://github.com/edsml-kl121/TH-azure-ai-workshop-2027/tree/main/LAB_6_observability)

Each lab is self-contained. Start with the `README.md` in its folder.
