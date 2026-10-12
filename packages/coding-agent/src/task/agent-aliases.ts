/**
 * Settings-defined task agent aliases (`task.agentAliases`): extra agent names
 * that reuse a discovered agent's definition, optionally with their own
 * task-list description and a provider `logit_bias` sent on every request.
 */
import { isRecord } from "@oh-my-pi/pi-utils";
import { MAIN_AGENT_RULE_NAME, SUB_AGENT_RULE_NAME } from "../capability/rule";
import type { AgentDefinition } from "./types";

/** One `task.agentAliases` entry. */
export interface AgentAlias {
	/** Name of the discovered agent whose definition the alias reuses. */
	agent: string;
	/** Task-list description; defaults to the aliased agent's. */
	description?: string;
	/** Sent as the provider's `logit_bias` on every request the alias makes. */
	logitBias?: Record<string, number>;
}

const ALIAS_FIELDS: Record<string, true> = { agent: true, description: true, logitBias: true };

function kindOf(value: unknown): string {
	return Array.isArray(value) ? "an array" : `a ${typeof value}`;
}

/**
 * Validate `task.agentAliases` (alias name → {@link AgentAlias}). An absent or
 * empty (`null`) mapping means no aliases, and a `null` alias or bias entry
 * clears one inherited from a lower-priority settings layer. Anything else
 * malformed fails settings load, so a misspelled field such as `logit_bias`
 * never silently drops the bias. The result is a null-prototype map, so any
 * alias name stays an own entry.
 */
export function validateAgentAliases(value: unknown): Record<string, AgentAlias> {
	const aliases: Record<string, AgentAlias> = Object.create(null);
	if (value === undefined || value === null) return aliases;
	if (!isRecord(value)) {
		throw new Error(`Invalid task.agentAliases: expected a map of alias name to alias, got ${kindOf(value)}.`);
	}
	for (const [name, entry] of Object.entries(value)) {
		if (entry === null) continue;
		const key = `task.agentAliases.${name}`;
		// `main` and `sub` are the rule-scoping sentinels; agent definitions reject them too.
		const lowered = name.toLowerCase();
		if (name === "" || name.trim() !== name || lowered === MAIN_AGENT_RULE_NAME || lowered === SUB_AGENT_RULE_NAME) {
			throw new Error(
				`Invalid ${key}: an alias name must be non-empty, have no surrounding whitespace, and not be "main" or "sub".`,
			);
		}
		if (!isRecord(entry)) {
			throw new Error(`Invalid ${key}: expected a map with an \`agent\` field, got ${kindOf(entry)}.`);
		}
		const unknownField = Object.keys(entry).find(field => !Object.hasOwn(ALIAS_FIELDS, field));
		if (unknownField !== undefined) {
			throw new Error(
				`Invalid ${key}: unknown field \`${unknownField}\`; expected \`agent\`, \`description\`, or \`logitBias\`.`,
			);
		}
		const { agent, description, logitBias } = entry;
		if (typeof agent !== "string" || agent.trim() === "") {
			throw new Error(`Invalid ${key}.agent: expected the name of the agent to alias.`);
		}
		if (description !== undefined && typeof description !== "string") {
			throw new Error(`Invalid ${key}.description: expected a string, got ${kindOf(description)}.`);
		}
		const alias: AgentAlias = { agent: agent.trim() };
		if (description !== undefined) alias.description = description;
		if (logitBias !== undefined && logitBias !== null) {
			if (!isRecord(logitBias)) {
				throw new Error(`Invalid ${key}.logitBias: expected a map of token to bias, got ${kindOf(logitBias)}.`);
			}
			const biases: [string, number][] = [];
			for (const [token, bias] of Object.entries(logitBias)) {
				if (bias === null) continue;
				if (typeof bias !== "number" || !Number.isFinite(bias)) {
					throw new Error(`Invalid ${key}.logitBias.${token}: expected a finite number, got ${String(bias)}.`);
				}
				biases.push([token, bias]);
			}
			if (biases.length > 0) alias.logitBias = Object.fromEntries(biases);
		}
		aliases[name] = alias;
	}
	return aliases;
}

/**
 * Append each alias as a renamed copy of the agent it names, carrying its own
 * description and logit bias. Aliases resolve only against `agents`: an alias
 * never shadows an agent of the same name and never targets another alias.
 */
export function withAgentAliases(
	agents: AgentDefinition[],
	aliases: Readonly<Record<string, AgentAlias>>,
): AgentDefinition[] {
	const added: AgentDefinition[] = [];
	for (const [name, alias] of Object.entries(aliases)) {
		const base = agents.find(agent => agent.name === alias.agent);
		if (!base || agents.some(agent => agent.name === name)) continue;
		added.push({
			...base,
			name,
			description: alias.description ?? base.description,
			logitBias: alias.logitBias,
			filePath: undefined,
		});
	}
	return added.length > 0 ? [...agents, ...added] : agents;
}
