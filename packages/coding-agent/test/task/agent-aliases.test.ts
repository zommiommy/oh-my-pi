import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { unregisterCustomApis } from "@oh-my-pi/pi-ai/api-registry";
import { createMockModel, registerMockApi } from "@oh-my-pi/pi-ai/providers/mock";
import { closeModelCache } from "@oh-my-pi/pi-catalog/model-cache";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { AgentStorage } from "@oh-my-pi/pi-coding-agent/session/agent-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { validateAgentAliases, withAgentAliases } from "@oh-my-pi/pi-coding-agent/task/agent-aliases";
import { runSubprocess } from "@oh-my-pi/pi-coding-agent/task/executor";
import type { AgentDefinition } from "@oh-my-pi/pi-coding-agent/task/types";
import { __resetDirsFromEnvForTests, removeWithRetries, setAgentDir } from "@oh-my-pi/pi-utils";
import { createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const WORKER: AgentDefinition = {
	name: "worker",
	description: "Worker",
	systemPrompt: "Do the work.",
	tools: ["read"],
	source: "bundled",
};

describe("task.agentAliases", () => {
	it("normalizes aliases and lets null clear an alias or a bias", () => {
		expect(validateAgentAliases(undefined)).toEqual({});
		expect(validateAgentAliases(null)).toEqual({});
		expect(
			validateAgentAliases({
				"scout-local": {
					agent: " scout ",
					description: "Local scout",
					logitBias: { "151667": -100, Paris: -5, cleared: null },
				},
				plain: { agent: "task", logitBias: { cleared: null } },
				removed: null,
			}),
		).toEqual({
			"scout-local": { agent: "scout", description: "Local scout", logitBias: { "151667": -100, Paris: -5 } },
			plain: { agent: "task" },
		});
	});

	it("rejects malformed aliases instead of silently dropping them", () => {
		const malformed: [unknown, string][] = [
			[[], "Invalid task.agentAliases:"],
			[{ main: { agent: "task" } }, "Invalid task.agentAliases.main:"],
			[{ " padded": { agent: "task" } }, "Invalid task.agentAliases. padded:"],
			[{ local: "scout" }, "Invalid task.agentAliases.local:"],
			[{ local: { logitBias: { "1": -100 } } }, "Invalid task.agentAliases.local.agent:"],
			[{ local: { agent: "scout", logit_bias: { "1": -100 } } }, "unknown field `logit_bias`"],
			[{ local: { agent: "scout", description: 42 } }, "Invalid task.agentAliases.local.description:"],
			// llama.cpp's array form is not the OpenAI map shape the providers send.
			[{ local: { agent: "scout", logitBias: [[1, -100]] } }, "Invalid task.agentAliases.local.logitBias:"],
			[{ local: { agent: "scout", logitBias: { "1": "-100" } } }, "Invalid task.agentAliases.local.logitBias.1:"],
			[{ local: { agent: "scout", logitBias: { "1": Number.POSITIVE_INFINITY } } }, "logitBias.1:"],
		];
		for (const [value, message] of malformed) {
			expect(() => validateAgentAliases(value)).toThrow(message);
		}
	});

	it("fails settings load on a malformed alias", async () => {
		await expect(
			Settings.loadIsolated({
				inMemory: true,
				overrides: { "task.agentAliases": { local: { agent: "scout", logit_bias: { "1": -100 } } } },
			}),
		).rejects.toThrow("unknown field `logit_bias`");
	});
});

describe("agent alias logit bias", () => {
	const MOCK_API_SOURCE = "test/agent-alias-logit-bias";
	const ENV_KEYS = ["HOME", "PI_CODING_AGENT_DIR", "OMP_PROFILE", "PI_PROFILE"] as const;
	let savedEnv: Record<string, string | undefined> = {};
	let root: string;
	let manager: AsyncJobManager;

	function restoreEnvValue(key: string, value: string | undefined): void {
		if (value === undefined) {
			delete process.env[key];
			delete Bun.env[key];
			return;
		}
		process.env[key] = value;
		Bun.env[key] = value;
	}

	beforeEach(async () => {
		savedEnv = Object.fromEntries(ENV_KEYS.map(key => [key, process.env[key]]));
		root = await fs.mkdtemp(path.join(os.tmpdir(), "omp-alias-logit-bias-"));
		const home = path.join(root, "home");
		await fs.mkdir(home, { recursive: true });
		restoreEnvValue("HOME", home);
		vi.spyOn(os, "homedir").mockReturnValue(home);
		setAgentDir(path.join(home, ".omp", "agent"));
		AgentRegistry.resetGlobalForTests();
		AgentLifecycleManager.resetGlobalForTests();
		registerMockApi(MOCK_API_SOURCE);
		manager = new AsyncJobManager({ maxRunningJobs: 4 });
		AsyncJobManager.setInstance(manager);
	});

	afterEach(async () => {
		await AgentLifecycleManager.global().dispose();
		AgentLifecycleManager.resetGlobalForTests();
		AgentRegistry.resetGlobalForTests();
		AsyncJobManager.setInstance(undefined);
		await manager.dispose({ timeoutMs: 1_000 });
		unregisterCustomApis(MOCK_API_SOURCE);
		vi.restoreAllMocks();
		for (const key of ENV_KEYS) restoreEnvValue(key, savedEnv[key]);
		__resetDirsFromEnvForTests();
		AgentStorage.close();
		closeModelCache();
		await removeWithRetries(root);
	});

	it("sends the alias logit bias on every turn of its subagent and persists it for revival", async () => {
		// Under the isolated HOME so project discovery never walks into the developer's real ~/.omp.
		const cwd = path.join(root, "home", "work");
		const artifactsDir = path.join(root, "artifacts");
		await fs.mkdir(cwd, { recursive: true });
		await fs.mkdir(artifactsDir, { recursive: true });
		const authStorage = createInMemoryAuthStorage();
		authStorage.keys.setRuntime("mock", "test-key");
		const modelRegistry = new ModelRegistry(authStorage);
		const mock = createMockModel({
			handler: context => {
				if (!(context.tools ?? []).some(tool => tool.name === "yield")) return { content: ["label"] };
				if (context.messages.at(-1)?.role === "toolResult") return { content: ["ok"] };
				return { content: [{ type: "toolCall", name: "yield", arguments: { type: "result", data: "done" } }] };
			},
		});
		const catalogAvailable = modelRegistry.getAvailable.bind(modelRegistry);
		vi.spyOn(modelRegistry, "getAvailable").mockImplementation(kind => [mock, ...catalogAvailable(kind)]);
		const logitBias = { "151667": -100, Paris: -100 };
		const alias = withAgentAliases(
			[WORKER],
			validateAgentAliases({ "biased-worker": { agent: "worker", logitBias } }),
		).find(agent => agent.name === "biased-worker");
		if (!alias) throw new Error("Expected the alias to resolve");

		try {
			const result = await runSubprocess({
				cwd,
				artifactsDir,
				agent: alias,
				task: "report",
				index: 0,
				id: "BiasedWorker",
				modelOverride: "mock/mock-model",
				authStorage,
				modelRegistry,
				settings: Settings.isolated({
					"compaction.enabled": false,
					"retry.enabled": false,
					"todo.enabled": false,
					"todo.reminders": false,
					"advisor.enabled": false,
					modelRoles: { default: "mock/mock-model" },
				}),
				enableLsp: false,
				enableMCP: false,
				enableIrc: false,
			});
			expect(result.exitCode).toBe(0);
		} finally {
			authStorage.close();
		}

		const turns = mock.calls.filter(call => (call.context.tools ?? []).some(tool => tool.name === "yield"));
		expect(turns.length).toBeGreaterThan(0);
		for (const turn of turns) expect(turn.options?.logitBias).toEqual(logitBias);
		const sessionFile = AgentRegistry.global().get("BiasedWorker")?.sessionFile;
		if (!sessionFile) throw new Error("Expected a persisted subagent session");
		expect((await SessionManager.peekSessionInit(sessionFile))?.init?.logitBias).toEqual(logitBias);
	}, 15_000);
});
