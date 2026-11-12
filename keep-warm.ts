import { isPlainObject, type Payload } from "./cached-request.ts";

const OUTPUT_LIMIT_KEYS = ["max_tokens", "max_output_tokens", "max_completion_tokens"] as const;

/** A warm-up that read less than this share of its prompt from the cache wrote it again instead, so warming stops. */
export const MIN_WARM_CACHED_SHARE = 0.9;

/**
 * How long before expiry the extension refreshes the cache. Pi's own warming refreshes at 90% of the lifetime (at least
 * 10 s before expiry), so the extension goes later and only acts once Pi has stopped, but still before compacting
 * before expiry (`compactLeadMs`), which is the fallback when a warm-up is not possible.
 */
export function warmLeadMs(lifetimeMs: number, compactLeadMs: number): number {
	return Math.max(compactLeadMs + 5_000, Math.floor(lifetimeMs * 0.05));
}

/**
 * Whether a cache expiring at `expiresAt` would cool before the keep-warm window ends: `keepWarmMinutes` after Pi
 * went idle at `idleSince`. Once a refresh carries the cache past the window, it is left to expire (or compacted).
 */
export function wantsWarm(expiresAt: number, idleSince: number, keepWarmMinutes: number): boolean {
	return keepWarmMinutes > 0 && expiresAt < idleSince + keepWarmMinutes * 60_000;
}

/**
 * Whether the captured request can be re-sent with a one-token output cap and still read its cache entry. Anthropic's
 * budget-based thinking (`thinking.type: "enabled"`) needs a budget below max_tokens, and Anthropic keys the message
 * cache on that budget; Pi's own warming skips these requests for the same reason. Adaptive thinking is fine.
 */
export function canReplay(captured: Payload): boolean {
	const thinking = captured.thinking;
	return !(isPlainObject(thinking) && thinking.type === "enabled");
}

/**
 * The warm-up payload: the captured request exactly as it was sent, with only its output limit taken from `params`,
 * which the provider built for maxTokens 1 (some APIs raise that to their minimum).
 */
export function warmPayload(params: unknown, captured: Payload): { payload: Payload } | { reason: string } {
	if (!isPlainObject(params)) return { reason: "the provider payload is not an object" };
	const payload: Payload = { ...captured };
	for (const key of OUTPUT_LIMIT_KEYS) {
		if (key in params) payload[key] = params[key];
		else delete payload[key];
	}
	return { payload };
}

/** The share of a response's prompt that was read from the cache. */
export function cachedShare(usage: { input: number; cacheRead: number; cacheWrite: number }): number {
	const prompt = usage.input + usage.cacheRead + usage.cacheWrite;
	return prompt > 0 ? usage.cacheRead / prompt : 0;
}

/** "off", "30 min", "1 h", "1.5 h". */
export function keepWarmLabel(minutes: number): string {
	if (minutes <= 0) return "off";
	return minutes < 60 ? `${minutes} min` : `${Number((minutes / 60).toFixed(1))} h`;
}
