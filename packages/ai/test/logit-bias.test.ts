import { describe, expect, it } from "bun:test";
import { streamSimple } from "@oh-my-pi/pi-ai/stream";
import type { Context, FetchImpl, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";

const context: Context = { messages: [{ role: "user", content: "ping", timestamp: 0 }] };
const logitBias = { "123": -100, "456": 25, " leading string": -5, "789": 0 };

async function requestBody(model: Model<"openai-completions" | "openai-responses">): Promise<Record<string, unknown>> {
	let captured: Record<string, unknown> | undefined;
	const fetchMock: FetchImpl = async (_url, init) => {
		if (typeof init?.body !== "string") throw new Error("Expected a JSON request body");
		captured = JSON.parse(init.body) as Record<string, unknown>;
		const event =
			model.api === "openai-completions"
				? { choices: [{ index: 0, delta: { content: "pong" }, finish_reason: "stop" }] }
				: {
						type: "response.completed",
						response: {
							status: "completed",
							usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
						},
					};
		return new Response(`data: ${JSON.stringify(event)}\n\ndata: [DONE]\n\n`, {
			headers: { "content-type": "text/event-stream" },
		});
	};
	await streamSimple(model, context, { apiKey: "test-key", fetch: fetchMock, logitBias }).result();
	if (!captured) throw new Error("Provider did not send a request");
	return captured;
}

function responsesModel(id: "gpt-4o-mini" | "gpt-5"): Model<"openai-responses"> {
	const model = getBundledModel<"openai-responses">("openai", id);
	if (!model) throw new Error(`Missing bundled model openai/${id}`);
	return model;
}

describe("logitBias request encoding", () => {
	it("encodes logit_bias with verbatim keys for Chat Completions even when penalties are unsupported", async () => {
		const bundled = responsesModel("gpt-4o-mini");
		const model = buildModel({
			...bundled,
			api: "openai-completions",
			compat: { ...bundled.compatConfig, supportsPenaltyAndStopParams: false },
		});
		expect(model.compat.supportsSamplingParams).toBe(true);
		expect((await requestBody(model)).logit_bias).toEqual(logitBias);
	});

	it("encodes logit_bias for Responses models that support sampling parameters", async () => {
		const model = responsesModel("gpt-4o-mini");
		expect(model.compat.supportsSamplingParams).toBe(true);
		expect((await requestBody(model)).logit_bias).toEqual(logitBias);
	});

	it("omits logit_bias for reasoning models that reject sampling parameters", async () => {
		const model = responsesModel("gpt-5");
		expect(model.compat.supportsSamplingParams).toBe(false);
		expect(await requestBody(model)).not.toHaveProperty("logit_bias");
	});
});
