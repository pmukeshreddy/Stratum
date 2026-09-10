# ChatGPT/Codex subscription transport

```text
Official app-server account/login + account/read + model/list
                     │ shared Codex credential store
Threadweave context → thin official-client bridge → subscription model
Threadweave actions ← structured Responses function calls + usage
```

Install the official Codex CLI and sign in using ChatGPT. Existing logins are reused.
`threadweave auth login --device` selects the official device-code flow. Status only
prints authentication type, plan when available, and client readiness—not tokens,
account email, or raw server/config responses. Status is a local credential check;
it does not guarantee that the next inference will pass account limits.

`threadweave auth install-client` builds the packaged thin Rust bridge against
official `openai/codex` revision `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`
(CLI 0.153.4). It requires git, Rust/cargo, network access and several GB of temporary
build space. The first build takes minutes; subsequent runs reuse the binary.
Build source/logs live under `~/.cache/threadweave` (or `XDG_CACHE_HOME`), outside
the repository, in a private directory. This is a pinned source integration, not
a promise of compatibility with arbitrary future Codex internal Rust APIs.

The bridge uses `codex-api::ResponsesClient`, `codex-login::AuthManager` and the
official auth-header provider. No token is sent over its stdio protocol. It uses
the subscription endpoint, not `api.openai.com`, strips API-key environment
variables, and refuses non-ChatGPT auth. It never invokes `codex exec`, starts a
Codex thread, reads AGENTS.md, loads Codex tools, or runs a Codex agent loop.
Parallel function calls become Threadweave actions. Inference processes can run
concurrently; a local lock coordinates refresh without serializing inference.

Codex's auth manager handles proactive refresh, same-account reload and 401 refresh
recovery for primary inference. Refinement's one-shot requests use Prime's
WebSocket-first transport, falling back to SSE only before streaming starts and
never on protocol/API errors. They bypass the SDK response parser and its 401
recovery loop, retaining original provider error codes, retry delays and terminal
statuses for Prime's shared completion retry policy. They have no session cache
key, reasoning option or extra transport retries. Initial credential resolution
still uses the official shared Codex store; missing credentials fail immediately.
Rejected refresh produces `AUTH_REQUIRED`, with no API fallback. Logout
removes/revokes the shared Codex credentials through the official account protocol.
Login/logout must not be used as a benchmark test against a human's active account.

Threadweave persists conversation pairs, context summaries and session IDs. Each
inference replays the active Threadweave context and uses its session ID as the
prompt-cache key. No provider conversation ID is required for recovery. Opaque
reasoning history is not replayed; exposed reasoning summaries/usage are recorded.
CLI run/eval snapshots pin the discovered model and effort. Direct Python users
can call `await threadweave.cli.resolved_config(config)` before session creation.

Streaming text, structured tool arguments, reported input/output/cached/reasoning
tokens, account-limit snapshots and refresh counts are retained. Cancellation and
timeouts terminate the inference process group. Runtime retry policy handles
transient inference errors. Monetary subscription cost is unavailable (`null`),
including recursive totals; no API dollar cost is inferred.

## Limits and security

The current backend does not accept a hard output-token cap. `max_output_tokens`
controls reservation and a client-visible byte guard, not server generation. A
response can consume more hidden/in-flight tokens than the reservation; total
limits are checked again at the next runtime boundary. Timeout/cancel accounting
is explicitly estimated. Temperature is not supported by this transport; unsupported
parameters fail rather than being silently dropped.

Managed account policy is checked through app-server before inference. Custom
Codex providers/API auth are refused. Keyring/file storage remains Codex-owned.
Local Python/process tools still have the host user's authority and could read
their files: do not run untrusted code under an account with valuable credentials.
The provider keeping tokens out of logs is not a sandbox for arbitrary host code.

## Live test

```sh
unset OPENAI_API_KEY
codex login status
uv run threadweave auth status
uv run threadweave doctor --config configs/coding.json
THREADWEAVE_LIVE_CODEX=1 THREADWEAVE_LIVE_DIRECTORY=results/subscription/live \
  uv run pytest -q -s tests/test_subscription_live.py
```

This opt-in test uses real subscription requests, repository search, Python state,
runtime restart, independent children and overlapping inference intervals. It is
an integration test, not a benchmark score. Ordinary tests use test-only protocol
fakes and make no model calls.

## Optional API provider

Only users intentionally choosing API billing should use this configuration:

```json
{
  "provider": {
    "name": "chat",
    "model": "YOUR_EXPLICIT_API_MODEL_ID",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

Set that variable in the daemon's environment and restart the daemon if changing
it. Subscription configs and benchmarks do not require or use it.

Official protocol references: [app-server](https://developers.openai.com/codex/app-server)
and [authentication](https://developers.openai.com/codex/auth).
