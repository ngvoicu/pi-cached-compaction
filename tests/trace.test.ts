import assert from "node:assert/strict";
import test from "node:test";
import { comparison, describeTrace, type TraceRecord, formatCacheStatus, formatRemaining, type CompactTrace, type TraceUsage } from "../trace.ts";

const cached = (usage: Partial<TraceUsage>, piEstimate: CompactTrace["piEstimate"]): CompactTrace => ({
	event: "compact",
	trigger: "manual",
	model: "m",
	outcome: "cached",
	tokensBefore: 0,
	cwd: "/",
	response: { stopReason: "stop", seconds: 1, summaryChars: 1, usage: { input: 0, cacheRead: 0, cacheWrite: 0, output: 0, cost: 0, ...usage } },
	piEstimate,
});

test("the card compares a cached compaction with Pi's: by cost when the model has prices, else by tokens not read from the cache", () => {
	assert.equal(
		comparison(cached({ cost: 0.061 }, { input: 186_000, cost: 0.47 }))?.text,
		"vs Pi's compaction: 87% less, $0.061 instead of ~$0.47 (it sends ~186k tokens, none from cache)",
	);
	assert.equal(
		comparison(cached({ input: 2, cacheRead: 39_559, cacheWrite: 457_879 }, { input: 186_000, cost: 0 }))?.text,
		"vs Pi's compaction: 2.5× as much, 458k tokens not from cache instead of ~186k",
	);
});

test("the log describes compacting when full", () => {
	const at = "2026-11-08T10:00:00.000Z";
	const time = new Date(at).toLocaleTimeString([], { hourCycle: "h23", hour: "2-digit", minute: "2-digit", second: "2-digit" });
	const base = { event: "full" as const, contextTokens: 905_000, threshold: 900_000, percent: 90, at, session: "s" };
	assert.equal(describeTrace({ ...base, action: "compact" }), `${time} context at 905k, past 90% (900k): compacting`);
	assert.equal(describeTrace({ ...base, action: "skip", reason: "Pi is busy or has queued messages" }), `${time} context at 905k, past 90% (900k): not compacting, Pi is busy or has queued messages`);
});

test("the footer labels the countdown as the cache's, and the /cached-compaction line keeps the bare time", () => {
	assert.equal(formatCacheStatus(59 * 60_000), "cache warm · 59m left");
	assert.equal(formatCacheStatus(120_000), "cache warm · 2m left");
	assert.equal(formatCacheStatus(25_000), "cache warm · 0:25 left");
	assert.equal(formatCacheStatus(400), "cache warm · 0:01 left");
	assert.equal(formatCacheStatus(0), "cache cold · next message full price");
	assert.equal(formatCacheStatus(-5_000), "cache cold · next message full price");
	assert.equal(formatRemaining(29 * 60_000), "29m");
	assert.equal(formatRemaining(25_000), "0:25");
});

test("the log describes keeping the cache warm", () => {
	const at = "2026-11-12T10:00:00.000Z";
	const time = new Date(at).toLocaleTimeString([], { hourCycle: "h23", hour: "2-digit", minute: "2-digit", second: "2-digit" });
	const base = { event: "warm" as const, secondsLeft: 180, contextTokens: 692_000, idleMinutes: 57, at, session: "s" };
	const usage = { input: 2, cacheRead: 692_000, cacheWrite: 0, output: 1, cost: 0.144 };
	assert.equal(describeTrace({ ...base, action: "warm", usage } as TraceRecord), `${time} kept the cache warm, idle 57 min: 99% of 692k from cache, 1 out, $0.14`);
	assert.equal(describeTrace({ ...base, action: "failed", reason: "overloaded" } as TraceRecord), `${time} keeping the cache warm failed: overloaded`);
	assert.equal(describeTrace({ ...base, action: "skip", reason: "Pi is busy or has queued messages" } as TraceRecord), `${time} cache expires in 180s: not keeping it warm, Pi is busy or has queued messages`);
});
