import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test, { mock } from "node:test";

// The trace path and settings are read from the agent directory, so it is set before the extension loads.
const dir = mkdtempSync(join(tmpdir(), "pcc-warm-"));
process.env.PI_CODING_AGENT_DIR = dir;
writeFileSync(join(dir, "settings.json"), JSON.stringify({ cachedCompaction: { keepWarmMinutes: 120, minContextTokens: 100_000 } }));
const { default: cachedCompaction } = await import("../index.ts");

const HOUR = 3_600_000;
const flush = () => new Promise((resolve) => setImmediate(resolve));

function harness() {
	const handlers = new Map<string, (event: unknown, ctx: unknown) => unknown>();
	const pi = {
		on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => handlers.set(event, handler),
		registerEntryRenderer: () => {},
		registerCommand: () => {},
		appendEntry: () => {},
		getThinkingLevel: () => "high",
	};
	cachedCompaction(pi as never);

	const model = { provider: "cliproxyapi", id: "sub/claude", api: "anthropic-messages", promptCache: { short: 3600, long: 3600 }, contextWindow: 1_000_000, maxTokens: 64_000 };
	const user = { role: "user", content: [{ type: "text", text: "hi" }], timestamp: 0 };
	const reply = { role: "assistant", content: [{ type: "text", text: "hello" }], timestamp: 1_000, stopReason: "stop", usage: { input: 2, output: 5, cacheRead: 600_000, cacheWrite: 0 } };
	const branch = [
		{ type: "message", id: "u", parentId: null, timestamp: new Date(0).toISOString(), message: user },
		{ type: "message", id: "a", parentId: "u", timestamp: new Date(1_000).toISOString(), message: reply },
	];
	const sent = { model: "claude", max_tokens: 64_000, system: "s", messages: [{ role: "user", content: "hi" }], thinking: { type: "adaptive" } };
	const warms: { payload: unknown; options: Record<string, unknown> }[] = [];
	let pending: ((value: unknown) => void) | undefined;
	const state = { compacts: 0, idle: true, hold: false };
	const ctx = {
		hasUI: false,
		cwd: dir,
		model,
		isIdle: () => state.idle,
		hasPendingMessages: () => false,
		isProjectTrusted: () => false,
		getContextUsage: () => ({ tokens: 600_000 }),
		sessionManager: { getBranch: () => branch, getSessionId: () => "s1", getSessionFile: () => undefined },
		modelRegistry: {
			streamSimple: (_model: unknown, _context: unknown, options: Record<string, unknown> & { onPayload: (p: unknown) => unknown; signal: AbortSignal }) => {
				warms.push({ payload: options.onPayload({ model: "rebuilt", max_tokens: 1, messages: [] }), options });
				const usage = { input: 2, output: 1, cacheRead: 600_000, cacheWrite: 0, cost: { total: 0.12 } };
				return {
					result: () =>
						state.hold
							? new Promise((resolve) => {
									pending = resolve;
									options.signal.addEventListener("abort", () => resolve({ stopReason: "aborted", usage }));
								})
							: Promise.resolve({ stopReason: "stop", usage }),
				};
			},
		},
		compact: () => state.compacts++,
	};
	const emit = (event: string, payload: unknown = {}) => handlers.get(event)?.(payload, ctx);
	// One real request: Pi builds the context, sends the payload, the reply lands, and the run settles.
	emit("context_with_system", { messages: [user] });
	emit("before_provider_request", { payload: sent });
	return { emit, warms, state, sent, release: () => pending?.({ stopReason: "stop", usage: { input: 2, output: 1, cacheRead: 600_000, cacheWrite: 0, cost: { total: 0.12 } } }) };
}

const traces = () =>
	readFileSync(join(dir, "cached-compaction.jsonl"), "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line))
		.filter((record) => record.event === "warm" || record.event === "expiry");

test("a 1-hour cache is refreshed twice inside a 2-hour idle window, then compacted just before it expires", async (t) => {
	mock.timers.enable({ apis: ["setInterval", "Date"], now: 1_000 });
	t.after(() => mock.timers.reset());
	const { emit, warms, state, sent } = harness();
	emit("agent_settled");

	// Pi's own warming would refresh at 54 min; the extension waits until 3 min before expiry (57 min).
	mock.timers.tick(HOUR - 181_000);
	assert.equal(warms.length, 0);
	mock.timers.tick(2_000);
	assert.equal(warms.length, 1);
	assert.deepEqual(warms[0].payload, { ...sent, max_tokens: 1 }, "the request is re-sent exactly as sent, with a one-token answer");
	assert.equal(warms[0].options.maxTokens, 1);
	assert.equal(warms[0].options.disableContinuation, true);
	await flush();

	// The refresh at ~57 min carries the cache to ~117 min, still inside the 121-min window: one more refresh at ~114 min.
	mock.timers.tick(HOUR - 181_000);
	assert.equal(warms.length, 1);
	mock.timers.tick(2_000);
	assert.equal(warms.length, 2);
	await flush();

	// That one carries it to ~174 min, past the window: no third refresh, and it compacts 20 s before expiry.
	mock.timers.tick(HOUR - 21_000);
	assert.equal(warms.length, 2);
	assert.equal(state.compacts, 0);
	mock.timers.tick(2_000);
	assert.equal(state.compacts, 1);
	assert.deepEqual(traces().map((r) => `${r.event}:${r.action}`), ["warm:warm", "warm:warm", "expiry:compact"]);
});

test("a new message aborts a refresh in flight, and nothing is recorded for it", async (t) => {
	mock.timers.enable({ apis: ["setInterval", "Date"], now: 1_000 });
	t.after(() => mock.timers.reset());
	writeFileSync(join(dir, "cached-compaction.jsonl"), "");
	const { emit, warms, state } = harness();
	state.hold = true;
	emit("agent_settled");
	mock.timers.tick(HOUR - 179_000);
	assert.equal(warms.length, 1);
	const signal = warms[0].options.signal as AbortSignal;
	state.idle = false;
	emit("agent_start");
	assert.equal(signal.aborted, true);
	await flush();
	assert.deepEqual(traces(), []);
});

test.after(() => {
	delete process.env.PI_CODING_AGENT_DIR;
	rmSync(dir, { recursive: true });
});
