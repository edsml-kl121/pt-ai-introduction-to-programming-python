---
name: python-refactor
description: Refactor existing Python code while preserving behavior. Use for cleanup, naming, function extraction, readability, or maintainability requests.
---

# Python refactoring

Follow this workflow:

1. Inspect every file directly involved in the requested change.
2. State the behavior that must remain unchanged.
3. Make one small refactoring at a time.
4. Prefer clear functions and names over new abstractions.
5. Preserve public function signatures unless the user explicitly requests an
   API change.
6. If tests exist, update them only when an intentional behavior change requires
   it. If no tests exist, recommend using the `python-testing` skill next.
7. Ask the `code-reviewer` subagent to check for accidental behavior changes.

End with a concise list of changed files and suggested local verification.
