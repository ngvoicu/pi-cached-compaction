import assert from "node:assert/strict";
import test from "node:test";
import { cachedShare, canReplay, keepWarmLabel, wantsWarm, warmLeadMs, warmPayload } from "../keep-warm.ts";

test("the extension refreshes after Pi's own warming would (10% of the lifetime) and before compacting before expiry", () => {
	assert.equal(warmLeadMs(3_600_000, 20_000), 180_000, "1-hour cache: 3 min before expiry; Pi refreshes at 6 min");
	assert.equal(warmLeadMs(1_800_000, 20_000), 90_000);
	assert.equal(warmLeadMs(300_000, 20_000), 25_000, "5-minute cache: 25 s before expiry; Pi refreshes at 30 s, compaction runs at 20 s");
});

test("the cache is kept warm only while it would otherwise cool inside the window after Pi went idle", () => {
	const idle = 0;
	assert.equal(wantsWarm(54 * 60_000, idle, 120), true);
	assert.equal(wantsWarm(119 * 60_000, idle, 120), true);
	assert.equal(wantsWarm(120 * 60_000, idle, 120), false, "the cache already lasts to the end of the window");
	assert.equal(wantsWarm(5 * 60_000, idle, 0), false, "off");
});

test("a request with Claude's budget-based thinking cannot be re-sent with a one-token answer; adaptive thinking and other APIs can", () => {
	assert.equal(canReplay({ model: "claude", thinking: { type: "enabled", budget_tokens: 8000 } }), false);
	assert.equal(canReplay({ model: "claude", thinking: { type: "adaptive" } }), true);
	assert.equal(canReplay({ model: "gpt", reasoning: { effort: "high" } }), true);
});

test("the warm-up re-sends the captured request unchanged except for the output limit the provider built", () => {
	const captured = { model: "claude", max_tokens: 64_000, system: "s", messages: [{ role: "user", content: "hi" }], thinking: { type: "adaptive" } };
	const built = warmPayload({ model: "other", max_tokens: 1, messages: [] }, captured);
	assert.deepEqual(built, { payload: { ...captured, max_tokens: 1 } });
	assert.equal(captured.max_tokens, 64_000, "the captured request is not changed");
	assert.deepEqual(warmPayload({ input: [] }, { input: ["x"], max_output_tokens: 32_000 }), { payload: { input: ["x"] } }, "a limit the provider leaves out is dropped");
	assert.deepEqual(warmPayload({ max_output_tokens: 16 }, { input: ["x"] }), { payload: { input: ["x"], max_output_tokens: 16 } });
	assert.deepEqual(warmPayload("nope", captured), { reason: "the provider payload is not an object" });
});

test("cached share and labels", () => {
	assert.equal(cachedShare({ input: 2, cacheRead: 692_000, cacheWrite: 0 }) > 0.99, true);
	assert.equal(cachedShare({ input: 0, cacheRead: 0, cacheWrite: 692_000 }), 0);
	assert.equal(cachedShare({ input: 0, cacheRead: 0, cacheWrite: 0 }), 0);
	assert.deepEqual([0, 30, 60, 90, 120, 480].map(keepWarmLabel), ["off", "30 min", "1 h", "1.5 h", "2 h", "8 h"]);
});
