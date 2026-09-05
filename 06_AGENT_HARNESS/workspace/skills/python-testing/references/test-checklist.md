# Test checklist

- Test public behavior rather than implementation details.
- Give each test one clear reason to fail.
- Include descriptive assertion messages only when they add useful context.
- Cover normal inputs before edge cases.
- Avoid network calls, timing dependencies, and shared mutable state.
- Use `python -m unittest discover` as the learner's run command.
