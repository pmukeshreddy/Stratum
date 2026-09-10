// Runs the unmodified Prime production SDK, kernel, reviewer and planner.
// Only the inference transport is adapted to the same real provider Buffalo uses.
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';

const [primeRoot, manifestPath, directory, python] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(manifestPath, 'utf8'));
const core = (p) => pathToFileURL(join(primeRoot, 'packages/coding-agent/src/core', p)).href;
const { createAgentSessionRuntime, createAgentSessionServices, createAgentSessionFromServices } = await import(core('agent-session-runtime.ts'));
const { SessionManager } = await import(core('session-manager.ts'));
const { SettingsManager } = await import(core('settings-manager.ts'));
const { getLocalHarnessStateDir } = await import(core('refinement/refinement.ts'));
const { createAssistantMessageEventStream } = await import(pathToFileURL(join(primeRoot, 'packages/ai/src/utils/event-stream.ts')).href);
const workspace = resolve(directory, 'workspace');
const agentDir = resolve(directory, 'agent');
mkdirSync(agentDir, { recursive: true });
writeFileSync(join(directory, 'host-requests.jsonl'), '', { flag: 'wx' });
writeFileSync(join(directory, 'driver-capabilities.json'), JSON.stringify({ native_continuation: true, host_request_observer: true, wait_for_descendants: true }));
let rootTurns = 0;
let rootId;
const identities = new Map();
// Opaque provider continuation never enters public traces or ordinary text.
const nativeContinuations = new Map();
const log = (file, data) => appendFileSync(join(directory, file), JSON.stringify(data) + '\n');

function transport(model, context, options = {}) {
  // Prime auxiliaries omit sessionId and are root-only; agent requests include it.
  const identity = identities.get(options.sessionId) ?? { id: rootId, depth: 0 };
  const stream = createAssistantMessageEventStream();
  const purpose = context.systemPrompt?.includes('automatic /refine review gate') ? 'refinement_review'
    : context.systemPrompt?.includes('/refine continual harness subsystem') ? 'refinement' : (context.tools?.length ? 'agent' : 'compaction');
  const messages = [{ role: 'system', content: context.systemPrompt ?? '' }];
  for (const m of context.messages) {
    const content = typeof m.content === 'string' ? m.content : m.content.filter(x => x.type === 'text').map(x => x.text).join('\n');
    if (m.role === 'toolResult') messages.push({ role: 'tool', tool_call_id: m.toolCallId, content });
    else if (m.role === 'assistant') messages.push({ role: 'assistant', content,
      ...(nativeContinuations.has(m.pairedContinuationId) ? { provider_items: nativeContinuations.get(m.pairedContinuationId), provider_identity: [spec.provider.name, spec.provider.model] } : {}),
      tool_calls: m.content.filter(x => x.type === 'toolCall').map(x => ({ id: x.id, type: 'function', function: { name: x.name, arguments: JSON.stringify(x.arguments) } })) });
    else messages.push({ role: m.role, content });
  }
  const request = {
    request_id: randomUUID(), session_id: identity.id, root_id: rootId, parent_id: identity.depth ? rootId : null,
    name: 'prime', turn: rootTurns, messages,
    tools: (context.tools ?? []).map(t => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } })),
    config: { ...spec.provider, max_output_tokens: options.maxTokens ?? spec.provider.max_output_tokens,
      parameters: { reasoning_effort: options.reasoning ?? 'none' } },
    request_kind: purpose.startsWith('refinement') ? 'auxiliary' : 'trajectory',
    reasoning_mode: purpose.startsWith('refinement') ? 'off' : 'inherit',
    input_token_bound: 0, metadata: { purpose, retain_provider_continuation: true },
  };
  const publicRequest = { ...request, messages: request.messages.map(({ provider_items, ...m }) => m) };
  log('provider-requests.jsonl', { time: Date.now(), request: publicRequest, prime_options: { reasoning: options.reasoning, maxTokens: options.maxTokens,
    native_continuation_messages: messages.filter(m => m.provider_items?.length).length } });
  const child = spawn(python, ['-m', 'threadweave.evals.prime_transport'], { cwd: spec.buffalo_source, stdio: ['pipe', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.on('data', b => { stdout += b; });
  child.stderr.on('data', b => { stderr += b; });
  const abort = () => child.kill('SIGTERM');
  options.signal?.addEventListener('abort', abort, { once: true });
  const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const message = { role: 'assistant', content: [], api: model.api, provider: model.provider, model: model.id, usage, stopReason: 'stop', timestamp: Date.now(), pairedContinuationId: request.request_id };
  stream.push({ type: 'start', partial: message });
  child.on('close', code => {
    options.signal?.removeEventListener('abort', abort);
    try {
      if (code !== 0) throw new Error(`Provider exited ${code}: ${stderr}`);
      const response = JSON.parse(stdout);
      if (response.provider_items?.length) nativeContinuations.set(message.pairedContinuationId, response.provider_items);
      delete response.provider_items;
      log('provider-calls.jsonl', { request: publicRequest, response });
      if (response.text) message.content.push({ type: 'text', text: response.text });
      for (const a of response.actions) message.content.push({ type: 'toolCall', id: a.id, name: a.name, arguments: a.arguments });
      Object.assign(usage, { input: response.usage.input_tokens - response.usage.cached_input_tokens,
        output: response.usage.output_tokens, cacheRead: response.usage.cached_input_tokens,
        totalTokens: response.usage.input_tokens + response.usage.output_tokens });
      message.stopReason = response.actions.length ? 'toolUse' : 'stop';
      for (const [contentIndex, part] of message.content.entries()) {
        stream.push(part.type === 'text' ? { type: 'text_end', contentIndex, content: part.text, partial: message }
          : { type: 'toolcall_end', contentIndex, toolCall: part, partial: message });
      }
      stream.push({ type: 'done', reason: message.stopReason, message });
    } catch (error) {
      message.stopReason = 'error'; message.errorMessage = String(error);
      log('transport-errors.jsonl', { error: String(error), request_id: request.request_id });
      stream.push({ type: 'error', reason: 'error', error: message });
    }
    stream.end();
  });
  child.on('error', error => { message.stopReason = 'error'; message.errorMessage = String(error); stream.push({ type: 'error', reason: 'error', error: message }); stream.end(); });
  child.stdin.end(JSON.stringify(request));
  return stream;
}

async function factory(options) {
  const identity = { id: options.sessionManager.getSessionId(), depth: options.sessionOptions?.rlmDepth ?? 0 };
  identities.set(identity.id, identity);
  if (identity.depth === 0) rootId = identity.id;
  if (identity.depth === 0 && spec.initial_harness_state) {
    const harnessDir = getLocalHarnessStateDir(options.sessionManager.getSessionArtifactDir());
    mkdirSync(harnessDir, { recursive: true });
    writeFileSync(join(harnessDir, 'harness_state.json'), JSON.stringify(spec.initial_harness_state));
  }
  const services = await createAgentSessionServices({ cwd: options.cwd, agentDir: options.agentDir,
    settingsManager: SettingsManager.inMemory({ telemetry: { enabled: false } }),
    telemetryDisabled: true, noBuiltinHerdrReporter: true,
    resourceLoaderOptions: { noContextFiles: true, noExtensions: true, appendSystemPrompt: [spec.system] } });
  services.modelRegistry.registerProvider('paired-real', {
    api: 'paired-real', apiKey: 'local-transport', baseUrl: 'local-stdio',
    streamSimple: transport,
    models: [{ id: spec.provider.model, name: spec.provider.model, reasoning: true,
      thinkingLevelMap: { xhigh: 'xhigh' }, input: ['text'], contextWindow: spec.context_tokens, maxTokens: spec.provider.max_output_tokens,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
  });
  const result = await createAgentSessionFromServices({ services, sessionManager: options.sessionManager,
    ...options.sessionOptions, model: services.modelRegistry.find('paired-real', spec.provider.model),
    thinkingLevel: spec.provider.parameters.reasoning_effort, serializedRefine: true,
    telemetryDisabled: true });
  // Observe only requests arriving from Prime's real kernel bridge. This never
  // originates a request and forwards each existing call exactly once, unchanged.
  result.session.handleRefineHostRequest = new Proxy(result.session.handleRefineHostRequest, {
    apply(target, receiver, args) {
      const response = Reflect.apply(target, receiver, args);
      log('host-requests.jsonl', { time: Date.now(), root_turn: rootTurns,
        session_id: identity.id, depth: identity.depth, operation: args[0], payload: args[1], response });
      return response;
    },
  });
  result.session.subscribe(event => {
    if (event.type === 'message_end' && event.message.role === 'assistant' && result.session.rlmDepth === 0) rootTurns++;
    log('events.jsonl', { time: Date.now(), root_turn: rootTurns, depth: result.session.rlmDepth, event });
  });
  return { ...result, services, diagnostics: services.diagnostics };
}

const runtime = await createAgentSessionRuntime(factory, { cwd: workspace, agentDir,
  sessionManager: SessionManager.create(workspace, join(directory, 'sessions')),
  sessionConfig: { serializedRefine: true, telemetryDisabled: true } });
writeFileSync(join(directory, 'initial-system-prompt.txt'), runtime.session.systemPrompt);
writeFileSync(join(directory, 'policy.json'), JSON.stringify(runtime.services.settingsManager.getAutoRefineSettings()));
try {
  for (const [index, stage] of spec.stages.entries()) {
    log('stages.jsonl', { index, state: 'started', root_turn: rootTurns, time: Date.now() });
    await runtime.session.prompt(stage.prompt);
    await runtime.session.waitForRlmQuiescence();
    writeFileSync(join(directory, 'conversation.json'), JSON.stringify(runtime.session.state.messages, null, 2));
    log('stages.jsonl', { index, state: 'completed', root_turn: rootTurns, time: Date.now() });
    process.stdout.write(JSON.stringify({ engine: 'prime', stage: index, root_turns: rootTurns }) + '\n');
  }
} finally {
  await runtime.dispose();
}
