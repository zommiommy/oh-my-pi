Manages context checkpoints for exploratory work so you can investigate broadly without permanently consuming context window.

**When to use:** Before any investigation that will require many tool calls (reading multiple files, grepping across directories, exploring unfamiliar code). Creating a checkpoint lets you compress the exploration into a concise report afterward, freeing context for the actual work.

**Actions:**
- `create` — mark current position before exploring (requires `goal` describing what you're investigating)
- `rewind` — erase all exploration since the checkpoint, keeping only your concise report (requires `report`)
- `drop` — discard the bookmark but keep the full exploration (use when the exploration turned out to be directly valuable)

**Rules:**
- You MUST close every checkpoint (rewind or drop) before yielding.
- Prefer `rewind` when the exploration was just information gathering. Prefer `drop` when the messages themselves are part of the deliverable.
- Checkpoints nest as a stack. Each rewind/drop pops the most recent one (DFS exploration).
- Call this tool alone — NEVER in parallel with itself.

**Typical flow:**
1. `checkpoint({ action: "create", goal: "..." })`
2. Perform exploratory work (read, grep, find, lsp, etc.)
3. `checkpoint({ action: "rewind", report: "..." })` with concise findings

After rewind, intermediate messages are replaced by your report — only the report survives in context.
