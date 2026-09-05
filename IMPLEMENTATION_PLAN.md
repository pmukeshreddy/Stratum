# Implementation plan

Repository inspection: the initial workspace contains only a six-byte README and
the supplied technical report. There are no existing implementations, dependencies,
provider interfaces, tests, hidden project instructions, or Git metadata to reuse.

1. Implement a Python package with validated configuration and durable SQLite
   sessions, append-only events, messages, goals, schedules, and versioned state.
2. Separate active context from per-session Python subprocesses and disk artifacts.
   Checkpoint recoverable values explicitly; report lost values and uncertain actions.
3. Implement an asynchronous session manager with model-controlled actions, recursive
   concurrent sessions, permissions, verifiers, budgets, refinement, and recovery.
4. Expose a local daemon and CLI, an offline deterministic demonstration, and a
   configurable streaming chat-completions provider behind an independent interface.
5. Exercise unit and integration tests, including a daemon killed and restarted in
   another process; document architecture, operation, extension, and limitations.

The report supplies architectural concepts only. All implementation names, code,
protocols, and package structure are original. No report source code is imported.
