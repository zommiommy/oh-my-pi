import type { AgentTool, AgentToolContext, AgentToolResult, AgentToolUpdateCallback } from "@oh-my-pi/pi-agent-core";
import { prompt } from "@oh-my-pi/pi-utils";
import { type Static, Type } from "@sinclair/typebox";
import checkpointDescription from "../prompts/tools/checkpoint.md" with { type: "text" };
import type { ToolSession } from ".";
import type { OutputMeta } from "./output-meta";
import { ToolError } from "./tool-errors";
import { toolResult } from "./tool-result";

// ── Payloads (consumed by session handler's tool_result / turn_end hooks) ──

/** Rewind: truncate history to checkpoint, inject report. */
export interface RewindPayload {
	readonly kind: "rewind";
	readonly checkpointsActive: number;
	readonly messageCount: number;
	readonly report: string;
}

/**
 * Drop: splice out the checkpoint call/result and the drop call/result,
 * preserving everything in between.
 */
export interface DropPayload {
	readonly kind: "drop";
	readonly checkpointsActive: number;
	readonly messageCount: number;
}

/**
 * Discriminated union consumed by the session handler via
 * {@link CheckpointController.consumePayload}.
 *
 * - `rewind`: truncate history to `messageCount`, inject `report`.
 * - `drop`: splice out checkpoint messages at `messageCount` (2 entries)
 *   and the drop messages at the end of history (2 entries).
 */
export type PendingPayload = RewindPayload | DropPayload;

// ── Checkpoint stack entry ────────────────────────────────────────────────

export interface CheckpointEntry {
	readonly goal: string;
	/**
	 * Message count captured AFTER the checkpoint tool result is appended.
	 * The session handler sets this via the controller after the tool_result
	 * event fires.
	 */
	readonly messageCount: number;
}

/**
 * Session-level checkpoint tracking state.
 * Used by the session handler to record where a checkpoint was created
 * in the message history and session tree.
 */
export interface CheckpointState {
	/** Number of in-memory messages at checkpoint (AFTER checkpoint tool result is appended) */
	checkpointMessageCount: number;
	/** Session entry ID at checkpoint (for session tree branching) */
	checkpointEntryId: string | null;
	/** Timestamp */
	startedAt: string;
}

/** Default maximum checkpoint nesting depth. */
export const DEFAULT_MAX_CHECKPOINT_DEPTH = 3;

// ── Controller ────────────────────────────────────────────────────────────

/**
 * Shared controller for checkpoint / rewind / drop.
 *
 * Checkpoints form a stack. Each `create` pushes; each `rewind` or `drop`
 * pops the most recent entry. This enables DFS-style exploration with
 * bounded nesting.
 *
 * The tool itself calls `activate(goal, 0)` — the session handler patches
 * the real `messageCount` after the tool_result event, because only the
 * handler has access to `agent.state.messages.length`.
 */
export class CheckpointController {
	readonly maxDepth: number;
	#stack: CheckpointEntry[] = [];
	#pendingPayload: PendingPayload | undefined;

	constructor(maxDepth: number = DEFAULT_MAX_CHECKPOINT_DEPTH) {
		this.maxDepth = maxDepth;
	}

	/** Whether at least one checkpoint is active. */
	get active(): boolean {
		return this.#stack.length > 0;
	}

	/** Current nesting depth (0 = no active checkpoints). */
	get depth(): number {
		return this.#stack.length;
	}

	/** The most recent (deepest) checkpoint entry, or undefined if inactive. */
	get state(): CheckpointEntry | undefined {
		return this.#stack.at(-1);
	}

	/**
	 * Push a new checkpoint onto the stack.
	 *
	 * Throws if max depth is reached, or if another checkpoint in the same
	 * turn already recorded the same messageCount (parallel tool calls).
	 *
	 * @param goal — what the agent is investigating
	 * @param messageCount — history length at checkpoint time; set to 0 by
	 *   the tool, then patched by the session handler after tool_result
	 */
	activate(goal: string, messageCount: number): void {
		if (this.#stack.length >= this.maxDepth) {
			throw new ToolError(
				`Maximum checkpoint depth (${this.maxDepth}) reached. Rewind or drop an existing checkpoint first.`,
			);
		}
		const top = this.#stack.at(-1);
		if (top && top.messageCount === messageCount && messageCount !== 0) {
			throw new ToolError(
				"Cannot create multiple checkpoints in the same turn. Checkpoint sequentially.",
			);
		}
		this.#stack.push({ goal, messageCount });
	}

	/**
	 * Update the messageCount of the most recent checkpoint.
	 * Called by the session handler after it observes the tool_result event
	 * and knows the actual agent message count.
	 */
	patchMessageCount(messageCount: number): void {
		const top = this.#stack.at(-1);
		if (!top) return;
		// Replace the top entry (entries are readonly interfaces, so rebuild)
		this.#stack[this.#stack.length - 1] = { ...top, messageCount };
	}

	/**
	 * Pop the deepest checkpoint and buffer a {@link RewindPayload}.
	 *
	 * Throws if the stack is empty, the report is empty, or a prior
	 * payload has not been consumed yet (parallel tool calls).
	 */
	rewind(report: string): RewindPayload {
		if (this.#pendingPayload) {
			throw new ToolError(
				"A rewind or drop is already pending. Cannot call rewind in the same turn.",
			);
		}
		const top = this.#stack.at(-1);
		if (!top) {
			throw new ToolError("No active checkpoint.");
		}
		const trimmed = report.trim();
		if (trimmed.length === 0) {
			throw new ToolError("Report cannot be empty.");
		}
		this.#stack.pop();
		const payload: RewindPayload = {
			kind: "rewind",
			checkpointsActive: this.#stack.length,
			messageCount: top.messageCount,
			report: trimmed,
		};
		this.#pendingPayload = payload;
		return payload;
	}

	/**
	 * Pop the deepest checkpoint and buffer a {@link DropPayload}.
	 * Keeps all exploration messages; the session handler splices out only
	 * the checkpoint and drop tool messages.
	 *
	 * Throws if the stack is empty or a payload is already pending.
	 */
	drop(): DropPayload {
		if (this.#pendingPayload) {
			throw new ToolError(
				"A rewind or drop is already pending. Cannot call drop in the same turn.",
			);
		}
		if (this.#stack.length === 0) {
			throw new ToolError("No active checkpoint to drop.");
		}
		const top = this.#stack.pop()!;
		const payload: DropPayload = {
			kind: "drop",
			checkpointsActive: this.#stack.length,
			messageCount: top.messageCount,
		};
		this.#pendingPayload = payload;
		return payload;
	}

	/**
	 * Consume the buffered payload. Returns undefined if no rewind or drop
	 * has occurred since the last consumption. Intended for session handler hooks.
	 */
	consumePayload(): PendingPayload | undefined {
		const payload = this.#pendingPayload;
		this.#pendingPayload = undefined;
		return payload;
	}

	/**
	 * Check whether the agent is about to yield with active checkpoints.
	 * Returns a warning message to inject if enforcement is needed,
	 * or undefined if no checkpoints are active.
	 *
	 * Call this from the agent's turn-end hook. If it returns a string,
	 * inject it as a system message and continue the run.
	 */
	enforceRewind(): string | undefined {
		if (this.#stack.length === 0) return undefined;
		const goals = this.#stack.map((s) => s.goal);
		return [
			"<system-interrupt>",
			`You have ${this.#stack.length} active checkpoint(s) that must be resolved before yielding.`,
			...goals.map((g) => `<checkpoint-goal>${g}</checkpoint-goal>`),
			"Call rewind (to erase exploration) or drop (to keep exploration) for each active checkpoint before finishing.",
			"</system-interrupt>",
		].join("\n");
	}

	/** Clear the entire checkpoint stack. Used on abort or cancel. */
	cancel(): void {
		this.#stack.length = 0;
		this.#pendingPayload = undefined;
	}
}

// ── Tool schema ───────────────────────────────────────────────────────────

const checkpointSchema = Type.Union(
	[
		Type.Object({
			action: Type.Literal("create"),
			goal: Type.String({ description: "What you are investigating and why" }),
		}),
		Type.Object({
			action: Type.Literal("rewind"),
			report: Type.String({ description: "Concise investigation findings to retain after rewind" }),
		}),
		Type.Object({
			action: Type.Literal("drop"),
		}),
	],
	{ discriminator: "action" },
);

type CheckpointParams = Static<typeof checkpointSchema>;

// ── Tool details ──────────────────────────────────────────────────────────

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

// ── Guards ────────────────────────────────────────────────────────────────

function isTopLevelSession(session: ToolSession): boolean {
	const depth = session.taskDepth;
	return depth === undefined || depth === 0;
}

// ── Tool ──────────────────────────────────────────────────────────────────

export class CheckpointTool implements AgentTool<typeof checkpointSchema, CheckpointToolDetails> {
	readonly name = "checkpoint";
	readonly label = "Checkpoint";
	readonly description: string;
	readonly parameters = checkpointSchema;
	readonly strict = true;

	constructor(
		private readonly session: ToolSession,
		private readonly controller: CheckpointController,
	) {
		this.description = prompt.render(checkpointDescription);
	}

	static createIf(session: ToolSession): CheckpointTool | null {
		if (!isTopLevelSession(session)) return null;
		const controller = session.checkpointController;
		if (!controller) return null;
		return new CheckpointTool(session, controller);
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
				// messageCount=0 is a placeholder; the session handler patches
				// it to the real value via controller.patchMessageCount() after
				// the tool_result event fires.
				this.controller.activate(params.goal, 0);
				const remaining = this.controller.maxDepth - this.controller.depth;
				return toolResult<CreateDetails>({
					action: "create",
					goal: params.goal,
					checkpointsRemaining: remaining,
				})
					.text(
						[
							"Checkpoint created.",
							`Goal: ${params.goal}`,
							`${remaining} more checkpoint(s) allowed.`,
							"Run your investigation, then call checkpoint with action rewind or drop.",
						].join("\n"),
					)
					.done();
			}
			case "rewind": {
				this.controller.rewind(params.report);
				return toolResult<RewindDetails>({
					action: "rewind",
					report: params.report,
				})
					.text(
						["Rewind requested.", "Report captured for context replacement."].join("\n"),
					)
					.done();
			}
			case "drop": {
				this.controller.drop();
				return toolResult<DropDetails>({ action: "drop" })
					.text("Checkpoint dropped. Exploration preserved.")
					.done();
			}
		}
	}
}
