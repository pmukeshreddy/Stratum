# Prime ARC protocol: source discovery

**BLOCKED: the published source does not establish the exact Figure 5 protocol. No new agent evaluation was launched.**

ARC evaluator: `398d4dd63cf01d00adbea41c13437ba0b8ad40fc` (`main`, 2026-08-12), [official repository](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent).

Published result metadata names Prime Agent **v0.3.3**, tag `bc6894c89577ad9bfcebca8055d155d9aaa27640`. The nearest main commit on the paper date is `a9b5d88b50db50860088772a408e867f8071b507`. Neither is proven to be the generating Figure 5 runtime; using the August 24 commit solely because of its date would be an assumption.

## Confirmed source behavior

The public broker uses ARC SDK 0.9.9 / arcengine 0.9.3 in OFFLINE mode with seed 0 and one retained environment per game. Published result data lists 25 versioned games, all matching our local corpus IDs. The agent-facing Python client exposes observe/status/act, a two-action bootstrap, and batches of at most 20. GAME_OVER causes one counted reset and ends the batch; a level change also ends the batch. WIN or ACTION_CAP stops broker mutations. The prompt prohibits delegation.

The executable action cap is 500, but 23 of the 75 published game records exceed it, reaching 5000. The launcher does not submit its RPC prompt, configure the terminal gate, or create the client filename/path referenced by the prompt. Its guard is never launched. Therefore the public launch path is not a complete reproducible account of the reported runs.

No Figure 5 checkpoint sequence, snapshot collector, plotter, interpolation rule, Best@1 calculation, RHAE exporter, or auxiliary-usage collector is present. The three published Opus runs do not establish whether points on a scaling curve use continuing trajectories or independent runs. Axis tick labels are not used as checkpoints.

## Execution gate

The previous independent controller is disabled. The older two-game validity run was gracefully cancelled and all artifacts retained. Exact protocol adaptation and both agent validity/full runs are blocked on the missing generating sources; production evaluator semantics were not changed to guessed defaults. The machine-readable spec and parity audit both set `full_run_permitted: false`.

## Thirty source-traced questions

### 1. Which environment/game set?

**Figure 5: UNSPECIFIED**

Published results contain the same 25 fully versioned public game IDs in each of three Opus runs. Broker selects the first local corpus environment whose ID starts with the requested prefix. No corpus file hashes or selection manifest are provided; Figure 5 correspondence is unproven.

Sources: [arc3_local_broker.py:40](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L40) — `Broker.__init__`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`; [pyproject.toml:6](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/pyproject.toml#L6) — `project.dependencies`.

### 2. Exactly how many games?

**Figure 5: UNSPECIFIED**

25 entries in each published run; run.sh launches ONE requested game and has no all-25 scheduler or task-count assertion.

Sources: [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`.

### 3. Seed/start-state behavior?

**Figure 5: UNSPECIFIED**

Broker creates Arcade(OFFLINE), calls make(resolved, seed=0, save_recording=True, include_frame_data=True), then env.reset(). Handle.actions starts at zero, so this explicit initial reset is not charged to its 500-action counter. No further trajectory initialization is in dispatch.

Sources: [arc3_local_broker.py:40](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L40) — `Broker.__init__`.

### 4. Root prompt?

**Figure 5: UNSPECIFIED**

Exact game-prompt.txt and AGENTS.md are embedded in this spec. They forbid RLM/agent creation, prescribe programmatic frame analysis and pre-action plans, and instruct continuation until WIN or ACTION_CAP. Prompt expects /workspace/fixed_broker_client.py; launcher only copies broker_client.py to a different workspace path. RPC startup does not submit its positional prompt.

Sources: [game-prompt.txt:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/game-prompt.txt#L1) — `initial prompt`; [AGENTS.md:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/AGENTS.md#L1) — `game behavioral guidance`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`; [packages/coding-agent/src/main.ts:1525](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/main.ts#L1525) — `main RPC dispatch`; [packages/coding-agent/src/modes/rpc/rpc-mode.ts:52](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/modes/rpc/rpc-mode.ts#L52) — `runRpcModeWithConnectionInternal`.

### 5. Environment tools/actions?

**Figure 5: UNSPECIFIED**

Client exposes observe(), status(), act(actions). Bootstrap requires observe, then a single non-RESET act, then a second single non-RESET act if still ACTIVE. Broker accepts RESET and ACTION1..7, batches 1..20, ACTION6 x/y integers 0..63. Observations include raw frames, state, levels_completed, available_actions, action count/remaining, terminal, and game_id (contrary to prompt description). Published broker does not return the batch trace promised by guidance.

Sources: [broker_client.py:31](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/broker_client.py#L31) — `call / _phase / _single_genuine`; [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [arc3_local_broker.py:55](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L55) — `Broker._observation`; [game-prompt.txt:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/game-prompt.txt#L1) — `initial prompt`; [AGENTS.md:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/AGENTS.md#L1) — `game behavioral guidance`.

### 6. What constitutes one agent turn?

**Figure 5: UNSPECIFIED**

runLoop emits turn_start, generates one assistant response, executes its tool-call batch, then emits turn_end. addAutonomousUsage increments turnsUsed at non-error assistant message_end. This is not an entire RPC prompt or an entire game.

Sources: [packages/agent/src/agent-loop.ts:307](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/agent/src/agent-loop.ts#L307) — `runLoop`; [packages/coding-agent/src/core/agent-session.ts:2973](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L2973) — `AgentSession._processAgentEvent / addAutonomousUsage`; [packages/coding-agent/src/core/autonomous.ts:171](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L171) — `addAutonomousUsage / autonomousTokenDelta`.

### 7. What causes another autonomous turn?

**Figure 5: UNSPECIFIED**

Tool results/steering continue the inner loop. When that loop would finish and no queued follow-ups exist, _getContinuationMessages calls nextAutonomousContinuation. With no gates and remaining limits it appends the generic autonomous user continuation. The provided run.sh sets --autonomous but no gate/limit overrides.

Sources: [packages/agent/src/agent-loop.ts:307](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/agent/src/agent-loop.ts#L307) — `runLoop`; [packages/coding-agent/src/core/agent-session.ts:2708](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L2708) — `AgentSession._getContinuationMessages`; [packages/coding-agent/src/core/autonomous.ts:196](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L196) — `nextAutonomousContinuation / shouldAutonomouslyContinue`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`.

### 8. Exact end-condition after a turn?

**Figure 5: UNSPECIFIED**

_shouldStopAfterTurn handles goal state, serialized refinement, and compaction. Autonomous quality gates run at continuation selection, not unconditionally after every tool-bearing turn. Client terminal-gate succeeds for WIN or ACTION_CAP, but run.sh never configures it. An ARC host terminal test for the reported evaluation is absent.

Sources: [packages/coding-agent/src/core/agent-session.ts:1729](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L1729) — `AgentSession._shouldStopAfterTurn`; [packages/coding-agent/src/core/autonomous.ts:196](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L196) — `nextAutonomousContinuation / shouldAutonomouslyContinue`; [broker_client.py:74](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/broker_client.py#L74) — `_main terminal-gate`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`.

### 9. What stops a game?

**Figure 5: UNSPECIFIED**

Broker revokes further mutations at WIN or ACTION_CAP. GAME_OVER triggers a counted reset if budget remains and flushes the batch. This does not itself terminate the agent process. Root prompt says stop at WIN/ACTION_CAP, but actual reported host stop wiring is missing.

Sources: [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [game-prompt.txt:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/game-prompt.txt#L1) — `initial prompt`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`; [packages/coding-agent/src/core/autonomous.ts:196](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L196) — `nextAutonomousContinuation / shouldAutonomouslyContinue`.

### 10. Action limits?

**Figure 5: UNSPECIFIED**

Executable broker cap is 500 cumulative actions, max 20 per call; each reset consumes one except initialization. A level-count change, GAME_OVER reset, WIN, or cap flushes the batch. Results contradict the 500 cap: 7/9/7 games across runs exceed it; maximum is 5000.

Sources: [arc3_local_broker.py:24](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L24) — `MAX_ACTIONS / MAX_BATCH`; [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`.

### 11. Token limits?

**Figure 5: UNSPECIFIED**

Launcher supplies none. Runtime v0.3.3 fallback autonomous limits: 3 continuations, 12 assistant turns, 80000 non-cache-read tokens, 1800000 ms. These bound autonomous continuation admission, not an actively generating tool loop. External/user settings and exact reported-run limits are missing.

Sources: [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`; [packages/coding-agent/src/core/autonomous.ts:48](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L48) — `DEFAULT_AUTONOMOUS_LIMITS / autonomousLimitReason`; [packages/agent/src/agent-loop.ts:307](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/agent/src/agent-loop.ts#L307) — `runLoop`.

### 12. Output tokens only or total?

**Figure 5: UNSPECIFIED**

Generic autonomous counter uses input + output + cacheWrite, excludes cacheRead. It is not an output-only scaling cap. getSessionStats totals input + output + cacheRead + cacheWrite. Figure 5 expenditure selection is not implemented in the published ARC code.

Sources: [packages/coding-agent/src/core/autonomous.ts:171](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L171) — `addAutonomousUsage / autonomousTokenDelta`; [packages/coding-agent/src/core/agent-session.ts:9045](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L9045) — `AgentSession.getSessionStats`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`.

### 13. Root + descendants?

**Figure 5: UNSPECIFIED**

ARC prompt explicitly forbids descendants. General Prime runtime can attribute child usage into a parent assistant entry, but that does not establish Figure 5 aggregation. The public ARC result export has no root/child breakdown or exporter.

Sources: [game-prompt.txt:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/game-prompt.txt#L1) — `initial prompt`; [AGENTS.md:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/AGENTS.md#L1) — `game behavioral guidance`; [packages/coding-agent/src/core/agent-session.ts:7320](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L7320) — `AgentSession._attributeRlmChildUsageToParent`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`.

### 14. Compaction/refinement/subagent calls counted?

**Figure 5: UNSPECIFIED**

Generic autonomous usage callback counts assistant message_end, not all provider calls. generateSummary and refinement helpers call completeSimple and return text/parsed edits, discarding response usage at those boundaries. A separate Figure 5 usage collector could differ, but none is published. Do not equate this counter with paper accounting.

Sources: [packages/coding-agent/src/core/autonomous.ts:171](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L171) — `addAutonomousUsage / autonomousTokenDelta`; [packages/coding-agent/src/core/agent-session.ts:2973](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L2973) — `AgentSession._processAgentEvent / addAutonomousUsage`; [packages/coding-agent/src/core/compaction/compaction.ts:556](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/compaction/compaction.ts#L556) — `generateSummary`; [packages/coding-agent/src/core/refinement/refinement.ts:841](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/refinement/refinement.ts#L841) — `planRefinement / reviewAutoRefine`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`.

### 15. Cumulative output thresholds per game?

**Figure 5: UNSPECIFIED**

No output-threshold array, threshold monitor, or per-snapshot output-token data exists in the ARC repository. results.json contains only total output tokens per full run.

Sources: [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`.

### 16. Figure 5 checkpoints from one trajectory?

**Figure 5: UNSPECIFIED**

No code establishes this. Broker and generic autonomous continuations retain state within one launch, but no curve collector links token thresholds to that state.

Sources: [arc3_local_broker.py:40](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L40) — `Broker.__init__`; [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [packages/agent/src/agent-loop.ts:307](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/agent/src/agent-loop.ts#L307) — `runLoop`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`.

### 17. Figure 5 independent fresh runs?

**Figure 5: UNSPECIFIED**

No code establishes this either. Three distinct Opus result objects and fresh_run_per_game=true are not evidence that each curve point is a fresh run.

Sources: [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`.

### 18. When/how is a checkpoint snapshotted?

**Figure 5: UNSPECIFIED**

No snapshot collector or RHAE-versus-token plotting implementation. The HTML only displays final scalar values from JSON.

Sources: [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`.

### 19. Can a response end while the game continues?

**Figure 5: UNSPECIFIED**

Yes in generic autonomous runtime: an ordinary final assistant response can be followed by another user continuation if gates/limits permit. Broker environment remains alive. End-to-end ARC wiring is not established because the launcher lacks its RPC driver.

Sources: [packages/agent/src/agent-loop.ts:307](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/agent/src/agent-loop.ts#L307) — `runLoop`; [packages/coding-agent/src/core/autonomous.ts:196](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L196) — `nextAutonomousContinuation / shouldAutonomouslyContinue`; [packages/coding-agent/src/core/agent-session.ts:2708](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L2708) — `AgentSession._getContinuationMessages`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`.

### 20. Next continuation prompt?

**Figure 5: UNSPECIFIED**

The generic DEFAULT_AUTONOMOUS_CONTINUATION_PROMPT is embedded below. Configured failed gates instead use buildAutonomousGateFailureContinuation with command, attempt, bounded output, timestamp. ARC launcher provides no custom continuation prompt or gate.

Sources: [packages/coding-agent/src/core/autonomous.ts:196](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/autonomous.ts#L196) — `nextAutonomousContinuation / shouldAutonomouslyContinue`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`.

### 21. Same model session survives?

**Figure 5: UNSPECIFIED**

Generic runLoop appends continuation messages to currentContext and invokes the same AgentSession. It does not create a new session at an autonomous continuation. No evidence identifies the Figure 5 checkpoint/session boundaries.

Sources: [packages/agent/src/agent-loop.ts:307](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/agent/src/agent-loop.ts#L307) — `runLoop`; [packages/coding-agent/src/core/agent-session.ts:2708](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L2708) — `AgentSession._getContinuationMessages`.

### 22. Same REPL survives?

**Figure 5: UNSPECIFIED**

AgentSession retains its IPython kernel provisioner across ordinary turns; continuation hooks do not rebuild it. Reload has separate replacement logic. ARC prompt explicitly directs retention in private IPython state. No checkpoint collector is available to prove that behavior across Figure 5 points.

Sources: [packages/coding-agent/src/core/agent-session.ts:6972](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L6972) — `AgentSession._ipythonKernelProvisioner`; [packages/coding-agent/src/core/agent-session.ts:2708](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L2708) — `AgentSession._getContinuationMessages`; [AGENTS.md:1](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/AGENTS.md#L1) — `game behavioral guidance`.

### 23. Same ARC environment survives?

**Figure 5: UNSPECIFIED**

Broker stores one Handle.env across all observe/status/act requests. It only resets in response to RESET or GAME_OVER recovery, within that same wrapper. Checkpoint lifecycle is unspecified.

Sources: [arc3_local_broker.py:40](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L40) — `Broker.__init__`; [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`.

### 24. Compaction during same trajectory?

**Figure 5: UNSPECIFIED**

General runtime supports in-session threshold compaction, preserves queued autonomous continuations, and resumes the same AgentSession. Published ARC launcher does not pin compaction/refinement settings and has no recorded trace of these events.

Sources: [packages/coding-agent/src/core/agent-session.ts:1729](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L1729) — `AgentSession._shouldStopAfterTurn`; [packages/coding-agent/src/core/agent-session.ts:2708](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L2708) — `AgentSession._getContinuationMessages`; [packages/coding-agent/src/core/compaction/compaction.ts:556](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/compaction/compaction.ts#L556) — `generateSummary`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`.

### 25. Completion before a later checkpoint?

**Figure 5: UNSPECIFIED**

Broker returns unchanged terminal observation on later act calls. There is no score carry-forward, early-completion cost treatment, or checkpoint aggregation code.

Sources: [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`.

### 26. Exact Best@1 semantics?

**Figure 5: UNSPECIFIED**

No executable selection/aggregation definition. results.json contains three full Opus runs and identifies run-1 as median; the viewer selects that record. Do not infer per-game best, best run, single attempt, or interpolation semantics from the name Best@1.

Sources: [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`.

### 27. RHAE aggregation?

**Figure 5: UNSPECIFIED**

No RHAE aggregation code is present. Viewer prints the stored rhae_percent scalar. Broker uses the official SDK internally, but never exports/closes/snapshots a scorecard for constructing the published scalar or curve.

Sources: [arc3_local_broker.py:40](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L40) — `Broker.__init__`; [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`; [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`.

### 28. Partial levels/progress?

**Figure 5: UNSPECIFIED**

Broker returns official observation.levels_completed and stops a batch whenever the count changes. Viewer displays levels_completed/total_levels. How partial levels enter Figure 5 RHAE, incomplete-game denominators, or missing-game handling is absent.

Sources: [arc3_local_broker.py:55](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L55) — `Broker._observation`; [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`.

### 29. Infrastructure failures/retries?

**Figure 5: UNSPECIFIED**

Broker catches exceptions and returns ok=false; client raises; launcher contains no retry loop. Guard source has fatal-ambiguity termination and read-only status timeout handling, but run.sh never starts that guard. Generic runtime retries transient provider errors and recovers context overflow in the same session; exact evaluation retry/usage-disposition settings are unpinned.

Sources: [arc3_local_broker.py:83](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/arc3_local_broker.py#L83) — `Broker.dispatch`; [broker_client.py:31](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/broker_client.py#L31) — `call / _phase / _single_genuine`; [game_protocol_guard.py:104](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/game_protocol_guard.py#L104) — `serve / _allowed / _forward`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`; [packages/coding-agent/src/core/agent-session.ts:8376](https://github.com/PrimeIntellect-ai/prime-agent/blob/bc6894c89577ad9bfcebca8055d155d9aaa27640/packages/coding-agent/src/core/agent-session.ts#L8376) — `AgentSession._handleRetryableError`.

### 30. Cost/output-token curve construction?

**Figure 5: UNSPECIFIED**

Missing collector, raw per-response/per-action trajectories, per-checkpoint data, model pricing manifest, interpolation/averaging rules, and plotting script. Only final Opus aggregate cost/output totals exist. No Sol/Terra/GLM Figure 5 result series are supplied.

Sources: [results/results.json:2](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/results.json#L2) — `protocol / runs / method`; [results/index.html:30](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/results/index.html#L30) — `render`; [run.sh:34](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent/blob/398d4dd63cf01d00adbea41c13437ba0b8ad40fc/run.sh#L34) — `top-level launcher`.

## Comparison with our evaluator

| Prime semantic | Our semantic | Match? |
|---|---|---|
| **game set**: 25 versioned IDs in published result data | Same 25 IDs in installed local corpus | MATCH — Exact ID-set equality checked locally; byte-for-byte match to paper corpus remains unspecified. |
| **environment version**: arc-agi 0.9.9; arcengine 0.9.3 | Installed arc-agi 0.9.9; arcengine 0.9.3 | MATCH — Installed dependency versions match the published pyproject. Prime corpus content hashes are absent. |
| **game lifecycle**: One OFFLINE Arcade/Handle per broker, seed 0, make then explicit reset | Independent official subprocess/Arcade per attempt, make(seed); no second reset | DIFFERENCE — Worker isolation matches; initialization call sequence differs. Two normalized observation hashes match despite that difference; retry lifecycle is also different. |
| **prompt**: Pinned game-prompt.txt + AGENTS.md, no delegation, pre-action PLAN | Generic benchmark_action task plus Buffalo tool instructions; delegation allowed | DIFFERENCE — Exact prompt text, guidance, and delegation constraints differ. Missing launcher prevents confirmation of the paper-effective prompt. |
| **environment interface**: Fixed Python observe/status/act, bootstrap, raw frames, batches <=20, GAME_OVER auto-reset | Dynamic benchmark_action(single action,data); raw frame plus RLE transport; explicit agent RESET | DIFFERENCE — Operation shape, bootstrap, batching, observations, and reset/level boundary handling differ. |
| **continuation**: Generic same-session continuation until configured gates/limits; actual ARC driver missing | Codex stops at turn/completed; Buffalo stops at Runtime session completion for ARC | DIFFERENCE — Our evaluator does not re-prompt an unsolved game after ordinary completion. The exact paper host gate and limits are unspecified. |
| **checkpointing**: UNSPECIFIED | Only final scorecard; obsolete independent sweep controller disabled | DIFFERENCE — No evidence for continuing-vs-independent points, thresholds, crossing boundaries, or carry-forward. No parity claim is possible. |
| **accounting**: Figure 5 UNSPECIFIED; generic autonomy input+output+cacheWrite, excludes cacheRead | All provider requests root+descendant+compaction+refinement, input/output recorded separately | DIFFERENCE — Generic counter is not paper expenditure evidence; silently adopting either would invent Figure 5 semantics. |
| **termination**: Broker terminal WIN/ACTION_CAP; GAME_OVER resets; host gate not wired | No host ARC-terminal stop; runner stops on agent completion or resource limits | DIFFERENCE — Broker end states and agent stop conditions differ; adding a guessed terminal driver cannot demonstrate exact parity. |
| **action limit**: 500 in broker, contradicted by published results | Uses run.limits.max_tool_calls; planned scaling config 10000 | DIFFERENCE — Budget values and counted entities differ. Prime paper-effective action budget requires its generating configuration. |
| **scoring**: UNSPECIFIED | Official EnvironmentScorecard.from_scorecard over disjoint final cards | DIFFERENCE — Our official scorer is concrete, but Prime exports only scalar results; its curve aggregation/Best@1 cannot be confirmed. |
| **retries**: No launcher retry loop; guard fatal ambiguity logic not launched; runtime provider retries configurable | Up to 3 fresh infrastructure-invalid attempts; failed usage retained separately | DIFFERENCE — Fresh-game retry behavior and accounting cannot be equated to the missing reported-run policy. |
| **persistence**: Generic session/context/kernel persist; broker Handle persists; no checkpoint lifecycle code | Same native thread/Runtime/Python/environment within attempt; fresh retry creates new state | DIFFERENCE — Within-attempt persistence exists in both. Persistence across curve points is unspecified and not implemented in our scheduler. |

## Validation performed

Four zero-inference control-flow probes of the unmodified published broker passed: exact 500-action cap, GAME_OVER reset and batch flush, level-change batch flush, and terminal WIN no further mutation. These are unit fixtures, not capability scores.

Two real official ARC environments were initialized twice with the published broker and once with our current initialization. Normalized observation hashes matched in each group. Installed SDK/engine versions match the public pin. This establishes initialization consistency for those two games only; it does not validate absent agent, continuation, checkpoint, or scoring semantics.

Evidence: [source probes](prime-protocol-evidence-20260908/source-probes.json), [initialization audit](prime-protocol-evidence-20260908/initialization-audit.json), [machine spec](prime-arc-protocol.json), [field-by-field parity audit](prime-protocol-parity.json).

## Exact remaining inputs

- `figure5_source_missing`: No Figure 5 runner, snapshot collector, plotting script, checkpoint sequence, or four-model raw trajectory data in either discovered repository.
- `action_budget_conflict`: Published executable cap is 500; 23 of 75 published game records exceed it, up to 5000 actions. The generating run configuration is absent.
- `launcher_incomplete`: run.sh selects RPC mode without an RPC prompt driver, omits terminal-gate configuration, references the wrong client filename/path, and never starts the published protocol guard.
- `accounting_and_scoring_export_missing`: No code links root/auxiliary usage, official scorecard snapshots, per-game aggregation, partial progress, retries, Best@1, or prices to Figure 5.
- `exact_run_identity_missing`: Public result metadata names Prime v0.3.3, but no immutable generating-run manifest proves a runtime SHA, environment byte hashes, effective settings, or Figure 5 provenance.

Required to proceed: the actual Figure 5 orchestration/plotting source and immutable run manifests, including action-budget overrides, RPC/terminal driver, effective runtime configuration, scorecard-to-token linkage, and usage/pricing extraction. A complete source link/path is sufficient; no new inference is needed to resolve this.
