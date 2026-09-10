# Domain-generality execution audit

The production path now selects a task adapter before admitting capabilities. `Runtime.create` loads extensions, `Environment.configure` resolves adapter defaults plus explicit grants minus disabled grants, and the registry enforces each capability's ownership. Coding tools are registered by `CodingAdapter.providers`, not by the generic built-ins. Internal `edit` and `context.focus` RPCs use the same admission checks.

`Runtime._kernel` passes admitted schemas and capability namespace factories to the worker. Generic bootstrap contains workspace, Python, shell, delegation, communication, observation, history, artifacts, state, skills, MCP and generic verification. The coding factory supplies repository, Git, editing, tests, build, lint, typecheck, benchmark and experiment bindings. Method tables are filtered against admitted schemas. Worker restore reconstructs these bindings before restoring user state.

`Context.messages` obtains specialization instructions and supplemental evidence through adapter/provider hooks. Generic instructions have no coding API, repository intelligence, test or worktree instructions. Coding retains its decision-support instructions, repository instruction discovery, focused evidence and repository intelligence. Coding evidence prioritization is supplied to generic history by its adapter. Refinement reads recent conversation directly.

`TaskConfig` owns generic specification, verifier, scope and adapter options. `CodingOptions` owns repository/revision, commands, benchmarks, baseline and test-protection settings. Legacy flat config files and stored configs load into `task.options.coding`; legacy session repository context and pending candidate admission metadata also migrate. Restart re-admits and persists effective capabilities before recovering actions.

RLM uses adapter-registered child profiles and lifecycle hooks. Coding owns candidate/test/performance/review profiles, Git worktree admission, baseline inheritance, patch accounting and recovery. Generic isolated workspaces are copies and need no Git repository. Each admission worker uses its own SQLite connection. Model/thinking overrides, inherited active tools, request provenance, observation and completion delivery remain on the existing runtime paths.

The generic runtime calls the prepare/verify contract. Coding retains baseline capture, independent test execution, test protection, mutation observation, evidence, candidate acceptance and failure retention. Disabling model-visible coding capabilities does not disable the coding verifier. Existing command/file verifiers and custom adapters remain supported.

Verification includes actual production Runtime sessions with captured provider requests; real worker execution and restart; shared and isolated children; tool and host-RPC rejection; explicit capability grants and disabled capabilities; persisted legacy config recovery; and a non-coding extension supplying its own tools, namespace and child profile. Existing tests retain their behavioral assertions; tests that deliberately exercise coding from a workspace now explicitly grant that capability.

The machine-readable Part A gate and test/build logs are under `results/domain-generality/`. ManyIH execution starts only after all gate entries pass.
