# free-model-aggregator-01

One server that serves OpenAI Chat Completions (`POST /v1/chat/completions`)
and Anthropic Messages (`POST /v1/messages`), in JSON and SSE streaming,
by routing each request to the best free model across ten simulated
providers on OpenAI, Anthropic and Gemini wires.

## Run

```sh
node main.ts            # trial mode: reads $MAKEFUN_PROMPT_FILE and $MAKEFUN_ENV_URL
node main.ts --list     # prints the catalog JSON and exits 0
```

Node 22 stdlib only; nothing is installed. TypeScript runs directly.

## Files

- `main.ts` — brief, catalog probe, register/poll, routing and failover, handlers
- `lib/providers.ts` — brief parsing, provider catalog probes (starter code, unchanged)
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation helpers (starter code, unchanged)
- `lib/translate.ts` — cross-wire request/response translation for all six client×upstream combinations
- `LICENSE` — MIT

## How it routes

For each request, every free model meeting the manifest's requirements is
a candidate, ordered by the quality table's `rating_milli` (highest
first), with providers that have already made at least their conservative
budget cap pushed below providers still under it. The caps are the
minimum budgets from the kit's sealed provider table (groq 9, cerebras 5,
gemini 6, zai 4, openrouter 10, sambanova 4, pollinations 10, cloudflare 7,
ovh 5, kilol 12) — a preference for spreading calls, never a refusal.
JSON-mode requests never go to an anthropic-wire provider, whose wire
cannot carry JSON mode. A 402, 429 or 5xx (or unreachable upstream) fails
over to the next candidate; a provider failing twice in a row is
blacklisted for the trial; if every candidate fails, the best candidates
are retried once (a scheduled failure is tied to the call number, so the
next call may succeed). Exactly one provider call succeeds per request.

Upstream answers are collected whole before anything is sent to the
client, so a failed upstream call never leaves a half-sent stream behind.
Translations preserve system, user text, tools, max_tokens, stops, images
and JSON mode exactly, and answers return in the client's wire with the
provider's usage counts and stop reason mapped to the same family.
No clock, no unseeded randomness: candidate order is a pure function of
the brief and the call counts so far.

## Practice scores (local runner)

| trial | total | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 25 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |

(Starter baseline on trial-0: 38 — quality 38, served 40, wire 39,
failover 0, catalog 100, 14 upstream calls, 8 of 20 valid.)
