Manages context checkpoints for exploratory work.

Actions:
  create — mark current position before exploring (requires goal)
  rewind — erase exploration, keep only a concise report (requires report)
  drop   — discard the bookmark, keep the full exploration

Each rewind/drop pops the most recent checkpoint (stack, DFS).
You MUST close every checkpoint (rewind or drop) before yielding.
Call this tool alone — NEVER in parallel with itself.
