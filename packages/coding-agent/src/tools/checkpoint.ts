/**
 * Checkpoint / Rewind / Drop — agent context compression via message filters.
 *
 * Instead of mutating message history, checkpoint operations leave markers
 * in the append-only history. A message filter (registered on the agent)
 * scans the history before each LLM call and computes the "visible view"
 * using balanced parenthesis matching:
 *
 *   create  = open paren  (2 messages: assistant tool call + tool result)
 *   rewind  = close paren → hide entire span, inject report as replacement
 *   drop    = close paren → hide only the create and drop markers, keep exploration
 *
 * Nesting is handled naturally: inner pairs are resolved first, outer pairs
 * see the already-filtered interior.
 *
 * ─── Data flow ──────────────────────────────────────────────────────────
 *
 *   1. Model calls checkpoint({ action: 'create', goal })
 *      → Tool appends assistant + toolResult messages to raw history.
 *        No state mutation. The depth counter increments for validation.
 *
 *   2. Model does exploratory work (many tool calls).
 *
 *   3a. Model calls checkpoint({ action: 'rewind', report })
 *       → Raw history now has: [create pair] [exploration] [rewind pair]
 *       → Next LLM call: filter hides the entire span, injects report.
 *       → Model sees: [...pre-checkpoint] [report message]
 *
 *   3b. Model calls checkpoint({ action: 'drop' })
 *       → Raw history now has: [create pair] [exploration] [drop pair]
 *       → Next LLM call: filter hides only the marker pairs.
 *       → Model sees: [...pre-checkpoint] [exploration]
 *
 * ─── Wiring ─────────────────────────────────────────────────────────────
 *
 *   agent.registerMessageFilter(checkpointFilter);
 */

import type { AgentTool, AgentToolContext, AgentToolResult, AgentToolUpdateCallback } from "@oh-my-pi/pi-agent-core";
import type { AgentMessage, MessageFilter } from "@oh-my-pi/pi-agent-core";
import { prompt } from "@oh-my-pi/pi-utils";
import { type Static, Type } from "@sinclair/typebox";
import checkpointDescription from "../prompts/tools/checkpoint.md" with { type: "text" };
import type { ToolSession } from ".";
import type { OutputMeta } from "./output-meta";
import { ToolError } from "./tool-errors";
import { toolResult } from "./tool-result";

// ── Depth tracker (validation only) ──────────────────────────────────────

/** Default maximum checkpoint nesting depth. */
export const DEFAULT_MAX_CHECKPOINT_DEPTH = 3;

/**
 * Lightweight tracker for tool-time validation. Counts open checkpoints
 * so the tool can reject at maxDepth or when no checkpoint is active.
 *
 * No message counts, no payloads, no hooks — all history manipulation
 * is handled by {@link checkpointFilter}.
 */
export class CheckpointDepthTracker {
	readonly maxDepth: number;
	#depth = 0;

	constructor(maxDepth: number = DEFAULT_MAX_CHECKPOINT_DEPTH) {
		this.maxDepth = maxDepth;
	}

	get active(): boolean {
		return this.#depth > 0;
	}
	get depth(): number {
		return this.#depth;
	}

	create(): void {
		if (this.#depth >= this.maxDepth) {
			throw new ToolError(
				`Maximum checkpoint depth (${this.maxDepth}) reached. Rewind or drop an existing checkpoint first.`,
			);
		}
		this.#depth++;
	}

	close(): void {
		if (this.#depth === 0) {
			throw new ToolError("No active checkpoint.");
		}
		this.#depth--;
	}

	/** Returns a warning string if checkpoints are active, undefined otherwise. */
	enforceRewind(): string | undefined {
		if (this.#depth === 0) return undefined;
		return [
			"<system-interrupt>",
			`You have ${this.#depth} active checkpoint(s) that must be resolved before yielding.`,
			"Call checkpoint with action rewind (to erase) or drop (to keep) for each before finishing.",
			"</system-interrupt>",
		].join("\n");
	}

	cancel(): void {
		this.#depth = 0;
	}
}

// ── Checkpoint message filter ────────────────────────────────────────────

/** Sentinel interface for the report message injected by the filter. */
interface CheckpointReportMessage {
	role: "developer";
	content: Array<{ type: "text"; text: string }>;
	attribution: "agent";
	timestamp: number;
}

/**
 * Extracts the checkpoint action from a message, if it is a checkpoint
 * tool result. Returns undefined for non-checkpoint messages.
 */
function getCheckpointAction(
	msg: AgentMessage,
): { action: "create" | "rewind" | "drop"; report?: string } | undefined {
	if (!("role" in msg) || msg.role !== "toolResult") return undefined;
	if (msg.toolName !== "checkpoint") return undefined;
	const details = msg.details as { action?: string; report?: string } | undefined;
	if (!details?.action) return undefined;
	if (details.action === "create" || details.action === "rewind" || details.action === "drop") {
		return { action: details.action, report: details.report };
	}
	return undefined;
}

/**
 * Balanced parenthesis filter for checkpoint/rewind/drop.
 *
 * Scans the message array for checkpoint tool results. Each `create` is an
 * open paren, each `rewind`/`drop` is a close paren. The filter resolves
 * matched pairs and hides/replaces messages accordingly:
 *
 * - **rewind**: hides the create pair (assistant + toolResult), all messages
 *   between, and the rewind pair. Injects the report as a developer message.
 * - **drop**: hides only the create pair and the drop pair. Exploration
 *   messages between them are preserved.
 *
 * Operates on toolResult messages (not assistant messages) to identify
 * checkpoint boundaries. Each toolResult at index `i` has a corresponding
 * assistant message at `i - 1`. Both are hidden together.
 */
export const checkpointFilter: MessageFilter = (messages) => {
	// Find all checkpoint toolResult indices and their actions.
	const markers: Array<{
		index: number;
		action: "create" | "rewind" | "drop";
		report?: string;
	}> = [];

	for (let i = 0; i < messages.length; i++) {
		const parsed = getCheckpointAction(messages[i]);
		if (parsed) {
			markers.push({ ...parsed, index: i });
		}
	}

	if (markers.length === 0) return messages;

	// Balanced parenthesis matching: build a set of indices to hide,
	// and a map of indices where report messages should be injected.
	const hidden = new Set<number>();
	const reportInjections = new Map<number, string>(); // index → report to inject BEFORE this index
	const stack: Array<{ index: number }> = []; // stack of open (create) marker indices

	for (const marker of markers) {
		if (marker.action === "create") {
			stack.push({ index: marker.index });
		} else {
			const open = stack.pop();
			if (!open) continue; // unmatched close — skip

			const openToolResultIdx = open.index;
			const closeToolResultIdx = marker.index;

			if (marker.action === "rewind") {
				// Hide everything from the create's assistant message through the
				// rewind's tool result (inclusive).
				for (let i = openToolResultIdx - 1; i <= closeToolResultIdx; i++) {
					if (i >= 0) hidden.add(i);
				}
				// Inject report after the hidden span.
				if (marker.report) {
					reportInjections.set(closeToolResultIdx, marker.report);
				}
			} else {
				// Drop: hide only the create pair (assistant + toolResult)
				// and the drop pair (assistant + toolResult).
				if (openToolResultIdx - 1 >= 0) hidden.add(openToolResultIdx - 1); // create assistant
				hidden.add(openToolResultIdx); // create toolResult
				if (closeToolResultIdx - 1 >= 0) hidden.add(closeToolResultIdx - 1); // drop assistant
				hidden.add(closeToolResultIdx); // drop toolResult
			}
		}
	}

	if (hidden.size === 0 && reportInjections.size === 0) return messages;

	// Build the filtered message array.
	const result: AgentMessage[] = [];
	for (let i = 0; i < messages.length; i++) {
		// If a report should be injected after a hidden rewind span, add it here.
		if (reportInjections.has(i)) {
			const report = reportInjections.get(i)!;
			result.push({
				role: "developer",
				content: [{ type: "text", text: report }],
				attribution: "agent",
				timestamp: Date.now(),
			} as CheckpointReportMessage as AgentMessage);
		}
		if (!hidden.has(i)) {
			result.push(messages[i]);
		}
	}

	return result;
};

// ── Tool schema ──────────────────────────────────────────────────────────

const checkpointSchema = Type.Object({
	action: Type.Unsafe<"create" | "rewind" | "drop">({
		type: "string",
		enum: ["create", "rewind", "drop"],
		description:
			"'create' to mark a checkpoint, 'rewind' to erase exploration and keep a report, 'drop' to discard the bookmark and keep exploration",
	}),
	goal: Type.Optional(Type.String({ description: "What you are investigating (required for create)" })),
	report: Type.Optional(
		Type.String({ description: "Concise investigation findings to retain (required for rewind)" }),
	),
});

type CheckpointParams = Static<typeof checkpointSchema>;

// ── Tool details ─────────────────────────────────────────────────────────

interface CreateDetails {
	action: "create";
	goal: string;
	checkpointsRemaining: number;
	meta?: OutputMeta;
}

interface RewindDetails {
	action: "rewind";
	report: string;
	meta?: OutputMeta;
}

interface DropDetails {
	action: "drop";
	meta?: OutputMeta;
}

export type CheckpointToolDetails = CreateDetails | RewindDetails | DropDetails;

// ── Guards ───────────────────────────────────────────────────────────────

function isTopLevelSession(session: ToolSession): boolean {
	const depth = session.taskDepth;
	return depth === undefined || depth === 0;
}

// ── Tool ─────────────────────────────────────────────────────────────────

export class CheckpointTool implements AgentTool<typeof checkpointSchema, CheckpointToolDetails> {
	readonly name = "checkpoint";
	readonly label = "Checkpoint";
	readonly description: string;
	readonly parameters = checkpointSchema;
	readonly strict = true;

	constructor(
		private readonly session: ToolSession,
		private readonly tracker: CheckpointDepthTracker,
	) {
		this.description = prompt.render(checkpointDescription);
	}

	static createIf(session: ToolSession): CheckpointTool | null {
		if (!isTopLevelSession(session)) return null;
		const tracker = session.checkpointTracker;
		if (!tracker) return null;
		return new CheckpointTool(session, tracker);
	}

	async execute(
		_toolCallId: string,
		params: CheckpointParams,
		_signal?: AbortSignal,
		_onUpdate?: AgentToolUpdateCallback<CheckpointToolDetails>,
		_context?: AgentToolContext,
	): Promise<AgentToolResult<CheckpointToolDetails>> {
		if (!isTopLevelSession(this.session)) {
			throw new ToolError("Checkpoint not available in subagents.");
		}

		switch (params.action) {
			case "create": {
				const goal = params.goal?.trim();
				if (!goal) {
					throw new ToolError("'goal' is required for action 'create'.");
				}
				this.tracker.create();
				const remaining = this.tracker.maxDepth - this.tracker.depth;
				return toolResult<CreateDetails>({
					action: "create",
					goal,
					checkpointsRemaining: remaining,
				})
					.text(
						[
							"Checkpoint created.",
							`Goal: ${goal}`,
							`${remaining} more checkpoint(s) allowed.`,
							"Run your investigation, then call checkpoint with action rewind or drop.",
						].join("\n"),
					)
					.done();
			}
			case "rewind": {
				const report = params.report?.trim();
				if (!report) {
					throw new ToolError("'report' is required for action 'rewind' and cannot be empty.");
				}
				this.tracker.close();
				return toolResult<RewindDetails>({
					action: "rewind",
					report,
				})
					.text(["Rewind requested.", "Report captured for context replacement."].join("\n"))
					.done();
			}
			case "drop": {
				this.tracker.close();
				return toolResult<DropDetails>({ action: "drop" })
					.text("Checkpoint dropped. Exploration preserved.")
					.done();
			}
			default:
				throw new ToolError(`Unknown checkpoint action: '${params.action}'. Use 'create', 'rewind', or 'drop'.`);
		}
	}
}
