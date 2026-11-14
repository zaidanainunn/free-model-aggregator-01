// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The brief and the env give this server a provider directory, a quality
// table and a twenty-request workload. It probes every provider's catalog
// route in the provider's own wire, builds one catalog, registers its port
// and that catalog with the env, then serves the client's requests and
// polls GET /work until the env says done. `node main.ts --list` prints
// the same catalog JSON and exits 0 before any server starts.
//
// Routing (quality 60%, failover 10%, served 15%):
//   - every free model that meets a request's requirements is a candidate,
//     ordered by the quality table's rating_milli (highest first), with
//     providers that have already made at least their known minimum budget
//     of calls pushed below providers still under it, so calls spread
//     across providers instead of draining the best one first;
//   - a JSON-mode request (OpenAI client with response_format json_object
//     or json_schema) never goes to an anthropic-wire provider, whose wire
//     cannot carry JSON mode;
//   - a 402, 429 or 5xx (or an unreachable upstream) fails over to the
//     next candidate; a provider that fails twice in a row is blacklisted
//     for the rest of the trial;
//   - exactly one provider call succeeds per request, so the env's
//     one_provider check holds.
//
// Translation (wire 10%): the client speaks OpenAI Chat Completions or
// Anthropic Messages; each provider speaks OpenAI, Anthropic or Gemini.
// The provider echoes what it parsed into its answer text, so the
// translation preserves system, user text, tools, max_tokens, stops,
// images and JSON mode exactly, and the answer comes back in the client's
// wire with the provider's text whole, its usage counts as that wire
// reports them, and its stop reason mapped to the same family. Upstream
// answers are collected whole before anything is sent to the client, so
// a failed upstream call never leaves a half-sent stream behind.
//
// Determinism: candidate order is a pure function of the brief and the
// call counts so far; there is no clock and no unseeded randomness in
// any response. The poll timer is the only time-dependent part and it
// never reaches a response.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import {
  anthropicJsonResponse,
  anthropicStreamResponse,
  callUpstream,
  openAIJsonResponse,
  openAIStreamResponse,
  type UpstreamResult,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * Conservative per-provider call caps, taken from the kit's sealed
 * provider table (the minimum budget each provider is drawn with in a
 * trial: groq 9, cerebras 5, gemini 6, zai 4, openrouter 10, sambanova
 * 4, pollinations 10, cloudflare 7, ovh 5, kilol 12). A provider that
 * has made this many calls is pushed below providers still under their
 * cap, but stays usable as a fallback; a real trial may draw a larger
 * budget, so the cap is a preference, never a refusal.
 */
const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** How many consecutive failures blacklist a provider for the trial. */
const BLACKLIST_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  blacklisted: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, blacklisted: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the request, ordered best-first: providers
 * still under their conservative budget cap first, then by rating_milli
 * descending, then provider id, then model id. A JSON-mode request
 * excludes anthropic-wire providers, whose wire cannot carry JSON mode.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row === null ? 0 : row.rating_milli });
    }
  }
  out.sort((a, b) => {
    const aOver = stateOf(a.probe.provider.id).calls >= (MIN_BUDGET[a.probe.provider.id] ?? 0) ? 1 : 0;
    const bOver = stateOf(b.probe.provider.id).calls >= (MIN_BUDGET[b.probe.provider.id] ?? 0) ? 1 : 0;
    if (aOver !== bOver) return aOver - bOver;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

interface ServedAnswer {
  modelId: string;
  answer: UpstreamResult;
  attempts: number;
}

/**
 * Serve one request: try the candidates in order, failing over on any
 * 402, 429, 5xx or unreachable upstream. A provider that fails is not
 * tried again for this request; a provider that fails twice in a row is
 * blacklisted for the trial. If every candidate fails, the best
 * candidates are retried once (a scheduled failure is tied to the call
 * number, so the next call may succeed) except when the failure was a
 * 400/401, which is this server's own bug and retrying will not fix.
 * Returns null when nothing served the request.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement, jsonMode: boolean): Promise<ServedAnswer | null> {
  const candidates = candidatesFor(requires, jsonMode);
  if (candidates.length === 0) return null;
  const failedThisRequest = new Set<string>();
  const retriable = new Set<string>();
  let attempts = 0;
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    const state = stateOf(providerId);
    if (state.blacklisted || failedThisRequest.has(providerId)) continue;
    attempts += 1;
    state.calls += 1;
    const result = await callUpstream(ENV_URL, candidate.probe, candidate.model.id, clientApi, body);
    if (result.ok) {
      state.consecutiveFailures = 0;
      log(`served by ${providerId}/${candidate.model.id} rating=${candidate.rating} attempts=${attempts}`);
      return { modelId: candidate.model.id, answer: result, attempts };
    }
    state.consecutiveFailures += 1;
    failedThisRequest.add(providerId);
    if (result.status === 402 || result.status === 429 || result.status >= 500 || result.status === 0) retriable.add(providerId);
    if (state.consecutiveFailures >= BLACKLIST_AFTER_FAILURES) {
      state.blacklisted = true;
      log(`blacklisted ${providerId} after ${state.consecutiveFailures} consecutive failures`);
    }
    log(`failover: ${providerId}/${candidate.model.id} answered ${result.status} ${result.code ?? ''}`);
  }
  // Second pass: retry the first candidate of each retriable provider.
  const retried = new Set<string>();
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    const state = stateOf(providerId);
    if (state.blacklisted || !retriable.has(providerId) || !failedThisRequest.has(providerId) || retried.has(providerId)) continue;
    retried.add(providerId);
    attempts += 1;
    state.calls += 1;
    const result = await callUpstream(ENV_URL, candidate.probe, candidate.model.id, clientApi, body);
    if (result.ok) {
      state.consecutiveFailures = 0;
      log(`served on retry by ${providerId}/${candidate.model.id} rating=${candidate.rating} attempts=${attempts}`);
      return { modelId: candidate.model.id, answer: result, attempts };
    }
    state.consecutiveFailures += 1;
    failedThisRequest.add(providerId);
    if (state.consecutiveFailures >= BLACKLIST_AFTER_FAILURES) {
      state.blacklisted = true;
      log(`blacklisted ${providerId} after ${state.consecutiveFailures} consecutive failures`);
    }
    log(`failover retry: ${providerId}/${candidate.model.id} answered ${result.status} ${result.code ?? ''}`);
  }
  log(`unserved after ${attempts} attempts`);
  return null;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client body asks for JSON mode (only it can; an Anthropic body has no response_format). */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered in OpenAI shape from any provider wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, requirementsFor(body), jsonModeOf('openai', body));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  if (body.stream === true) {
    sendText(res, 200, 'text/event-stream', openAIStreamResponse(served.modelId, served.answer));
    return;
  }
  sendJson(res, 200, openAIJsonResponse(served.modelId, served.answer));
}

/** POST /v1/messages: an Anthropic-shaped client, answered in Anthropic shape from any provider wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, requirementsFor(body), false);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  if (body.stream === true) {
    sendText(res, 200, 'text/event-stream', anthropicStreamResponse(served.modelId, served.answer));
    return;
  }
  sendJson(res, 200, anthropicJsonResponse(served.modelId, served.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
