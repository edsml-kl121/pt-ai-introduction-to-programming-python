---
name: python-testing
description: Add or improve focused Python unit tests. Use for requests about tests, test cases, edge cases, unittest, or verifying Python behavior.
---

# Python testing

Follow this workflow:

1. Read the Python module that needs tests.
2. Read `references/test-checklist.md`.
3. Identify normal behavior, one edge case, and one error case when applicable.
4. Add tests using Python's built-in `unittest` module so the sample project
   needs no additional test dependency.
5. Do not change application behavior merely to make a weak test pass.
6. Ask the `code-reviewer` subagent to review test coverage and assertions.
7. Explain which command the learner should run because this agent has no shell
   execution tool.

Keep the test file small and readable.
