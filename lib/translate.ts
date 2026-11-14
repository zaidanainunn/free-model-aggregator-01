// Cross-wire translation for free-model-aggregator-01.
//
// The trial's client speaks OpenAI Chat Completions or Anthropic Messages.
// Each provider speaks its own wire (OpenAI, Anthropic or Gemini). The
// provider echoes what it parsed (system, user text, tools, max_tokens,
// stops, image, JSON mode) into its answer text, and the env's checks
// compare that echo with what the client sent. So every translation here
// must preserve those fields exactly, in the upstream wire's own spelling,
// and every answer must be returned in the client's wire with the
// provider's text whole, its usage counts as that wire reports them, and
// its stop reason mapped to the same family.
//
// Everything here is a pure function of its arguments except the upstream
// call itself: no clock, no randomness. Upstream answers are collected
// whole before anything is sent to the client, so a failed upstream call
// never leaves a half-sent stream behind for failover to clean up.

import { authHeaders, isRecord, joinBase, type Json, type Probe } from './providers.ts';
import { anthropicToOpenAI } from './wire.ts';

export interface Usage {
  input: number;
  output: number;
}

export interface UpstreamResult {
  ok: boolean;
  status: number;
  code: string | null;
  text: string;
  usage: Usage;
  /** The provider's stop token: "stop" | "end_turn" | "STOP" (family 0 in practice). */
  stop: string;
}

const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** OpenAI finish_reason for a provider stop token (stop families: 0 stop, 1 length, 2 tool, 3 refusal). */
export function openAIFinish(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Anthropic stop_reason for a provider stop token. */
export function anthropicStop(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Client -> upstream request bodies
// ---------------------------------------------------------------------------

/** OpenAI Chat Completions body as an Anthropic Messages body naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'assistant') {
      const content = raw.content;
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        try {
          const parsed: unknown = typeof call.function.arguments === 'string' ? JSON.parse(call.function.arguments) : call.function.arguments;
          if (isRecord(parsed)) input = parsed;
        } catch {
          input = {};
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    // user message
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks: Json[] = [];
      let hasImage = false;
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          hasImage = true;
          const url = block.image_url.url;
          const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
          if (m !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
      messages.push({ role: 'user', content: hasImage ? blocks : textOf(raw.content) });
      continue;
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
  fileData?: { fileUri: string };
  functionCall?: { name: string; args: Json };
  functionResponse?: { name: string; response: Json };
}

function openAIContentParts(content: unknown): GeminiPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: GeminiPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
      if (m !== null) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

function anthropicContentParts(content: unknown): GeminiPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: GeminiPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') parts.push({ fileData: { fileUri: block.source.url } });
    } else if (block.type === 'tool_use') parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
    else if (block.type === 'tool_result') {
      parts.push({ functionResponse: { name: '', response: { content: typeof block.content === 'string' ? block.content : textOf(block.content) } } });
    }
  }
  return parts;
}

/** Any client body (OpenAI or Anthropic) as a Gemini generateContent body naming `model` (model goes in the path). */
export function clientToGeminiRequest(clientApi: 'openai' | 'anthropic', body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const declarations: Json[] = [];
  let maxTokens: number | undefined;
  let stops: string[] = [];
  let jsonMode = false;
  let temperature: number | undefined;
  let topP: number | undefined;

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const t = textOf(raw.content);
        if (t !== '') systemParts.push(t);
        continue;
      }
      if (raw.role === 'assistant') {
        const parts = openAIContentParts(raw.content);
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let args: Json = {};
          try {
            const parsed: unknown = typeof call.function.arguments === 'string' ? JSON.parse(call.function.arguments) : call.function.arguments;
            if (isRecord(parsed)) args = parsed;
          } catch {
            args = {};
          }
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role === 'tool') {
        contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }] });
        continue;
      }
      const parts = openAIContentParts(raw.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        declarations.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
      }
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    else if (typeof body.max_completion_tokens === 'number') maxTokens = body.max_completion_tokens;
    if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) jsonMode = true;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
  } else {
    const systemText = textOf(body.system);
    if (systemText !== '') systemParts.push(systemText);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const parts = anthropicContentParts(raw.content);
      if (parts.length === 0) continue;
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        declarations.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
      }
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream response parsing (collected whole)
// ---------------------------------------------------------------------------

interface SseFrame {
  event: string | null;
  data: string;
}

function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const raw of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const parts: string[] = [];
  for (const c of candidates) {
    if (!isRecord(c) || !isRecord(c.content) || !Array.isArray(c.content.parts)) continue;
    for (const p of c.content.parts) if (isRecord(p) && typeof p.text === 'string') parts.push(p.text);
  }
  return parts.join('');
}

function geminiUsageOf(body: Json): Usage {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

function geminiStopOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return typeof first.finishReason === 'string' ? first.finishReason : 'STOP';
}

/** Parse an upstream answer (JSON or SSE) into text, usage and stop. */
function parseUpstreamAnswer(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): { text: string; usage: Usage; stop: string } {
  if (!stream) {
    const body = parseJson(text);
    if (!isRecord(body)) return { text: '', usage: { input: 0, output: 0 }, stop: 'stop' };
    if (wire === 'openai') {
      const choices = Array.isArray(body.choices) ? body.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const message = isRecord(choice.message) ? choice.message : {};
      const t = typeof message.content === 'string' ? message.content : textOf(message.content);
      const usage = isRecord(body.usage) ? body.usage : {};
      return { text: t, usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) }, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' };
    }
    if (wire === 'anthropic') {
      const t = textOf(body.content);
      const usage = isRecord(body.usage) ? body.usage : {};
      return { text: t, usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) }, stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn' };
    }
    return { text: geminiTextOf(body), usage: geminiUsageOf(body), stop: geminiStopOf(body) };
  }
  // SSE
  const frames = parseSse(text);
  let outText = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'stop';
  for (const frame of frames) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = parseJson(frame.data);
    if (!isRecord(chunk)) continue;
    if (wire === 'openai') {
      if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') outText += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    } else if (wire === 'anthropic') {
      if (frame.event === 'message_start' && isRecord(chunk.message) && isRecord(chunk.message.usage)) usage = { ...usage, input: numberOrZero(chunk.message.usage.input_tokens) };
      if (frame.event === 'content_block_delta' && isRecord(chunk.delta) && chunk.delta.type === 'text_delta' && typeof chunk.delta.text === 'string') outText += chunk.delta.text;
      if (frame.event === 'message_delta') {
        if (isRecord(chunk.delta) && typeof chunk.delta.stop_reason === 'string') stop = chunk.delta.stop_reason;
        if (isRecord(chunk.usage)) usage = { ...usage, output: numberOrZero(chunk.usage.output_tokens) };
      }
    } else {
      outText += geminiTextOf(chunk);
      const u = geminiUsageOf(chunk);
      if (u.input !== 0 || u.output !== 0) usage = u;
      const s = geminiStopOf(chunk);
      if (s !== 'STOP' || stop === 'stop') stop = s;
    }
  }
  if (wire === 'gemini' && stop === 'stop') stop = 'STOP';
  return { text: outText, usage, stop };
}

function errorCodeOf(wire: string, status: number, text: string): string {
  const body = parseJson(text);
  if (isRecord(body)) {
    const err = isRecord(body.error) ? body.error : body;
    if (typeof err.code === 'string') return err.code;
    if (typeof err.type === 'string') return err.type;
  }
  return `http_${status}`;
}

// ---------------------------------------------------------------------------
// One upstream call
// ---------------------------------------------------------------------------

/**
 * Call one provider with the client's request translated to its wire.
 * Returns ok:false on any non-2xx (the caller fails over to the next
 * candidate); the response is collected whole so a stream is never
 * half-sent to the client before a failure is known.
 */
export async function callUpstream(
  envUrl: string,
  probe: Probe,
  modelId: string,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<UpstreamResult> {
  const provider = probe.provider;
  const base = joinBase(envUrl, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url = '';
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientApi === 'openai') {
      upstreamBody = { ...body, model: modelId };
      if (stream) upstreamBody.stream_options = { include_usage: true };
    } else {
      upstreamBody = anthropicToOpenAI(body, modelId);
      if (stream) upstreamBody.stream_options = { include_usage: true };
    }
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'anthropic') upstreamBody = { ...body, model: modelId };
    else upstreamBody = openAIToAnthropicRequest(body, modelId);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const encoded = encodeURIComponent(modelId);
    url = stream ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${encoded}:generateContent`;
    upstreamBody = clientToGeminiRequest(clientApi, body);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(10000) });
  } catch (err) {
    return { ok: false, status: 0, code: 'upstream_unreachable', text: '', usage: { input: 0, output: 0 }, stop: 'stop' };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, code: errorCodeOf(provider.wire, res.status, text), text: '', usage: { input: 0, output: 0 }, stop: 'stop' };
  const parsed = parseUpstreamAnswer(provider.wire, stream, text);
  return { ok: true, status: res.status, code: null, ...parsed };
}

// ---------------------------------------------------------------------------
// Client responses (from a collected upstream answer)
// ---------------------------------------------------------------------------

export function openAIJsonResponse(model: string, answer: UpstreamResult): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIFinish(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

export function openAIStreamResponse(model: string, answer: UpstreamResult): string {
  const finish = openAIFinish(answer.stop);
  const chunk = (delta: Json, fin: string | null): string => `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: fin }] })}\n\n`;
  let out = chunk({ role: 'assistant', content: '' }, null);
  // One content delta carrying the whole text is a valid stream; split it
  // into 13-char pieces like the provider does, for the same shape.
  for (let at = 0; at < answer.text.length; at += 13) out += chunk({ content: answer.text.slice(at, at + 13) }, null);
  out += chunk({}, finish);
  out += `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } })}\n\n`;
  out += 'data: [DONE]\n\n';
  return out;
}

export function anthropicJsonResponse(model: string, answer: UpstreamResult): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

export function anthropicStreamResponse(model: string, answer: UpstreamResult): string {
  const event = (name: string, data: unknown): string => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  let out = event('message_start', {
    type: 'message_start',
    message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } },
  });
  out += event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (let at = 0; at < answer.text.length; at += 13) {
    out += event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text.slice(at, at + 13) } });
  }
  out += event('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += event('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.usage.output } });
  out += event('message_stop', { type: 'message_stop' });
  return out;
}
