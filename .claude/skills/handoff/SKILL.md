---
name: handoff
description: Produce a verified session handover, reconcile current documentation and record unfinished work. Use for /handoff, session close-out or preparing work for the next agent.
---

# Handoff

Make it possible to resume from repository evidence rather than conversation
memory. Adapting or inspecting this skill does not invoke its workflow.

## Inventory and reconcile

1. Inspect the worktree and available history. Verify changed files, decisions,
   measurements and failures against actual artifacts. Preserve unrelated changes.
2. Revalidate running jobs and listeners. Record command, handle/PID, checkpoint,
   receipt, last observed progress and a way to check them again. Leave authorized
   ongoing work running; handoff does not imply cancellation or completion.
3. Update each existing document that owns a changed fact. Replace superseded
   current-state claims and keep dated experimental evidence intact. Reference
   measured artifacts rather than duplicating extensive numerical tables.
4. Record actual user decisions and authorization, remaining questions and known
   traps. Use the project's existing record/ledger if it has one. Do not introduce
   another documentation hierarchy merely to imitate the source repository.

For Bob, use the [project map](../../../docs/agent-workflows.md). For other
repositories, discover equivalent documents, validation and resource constraints.

## Write the next-agent brief

Use the existing handover location, or `docs/NEXT_AGENT_BRIEF.md` when none exists.
Keep it compact and make every volatile fact a dated observation with a recheck.

```markdown
# Next agent brief

Written: <timestamp and timezone>; repository state: <HEAD or no commits>, <branch>

## Goal and standing instructions
The original requested outcome, accepted choices and operational constraints.

## Verify first
| Fact observed at handoff | Evidence/handle | Recheck command or tool |
|---|---|---|

## What changed and what was measured
Source/artifact links, validation scope, failures and unresolved limitations.

## Next work
The next action, its dependency and the evidence that would establish success.

## Running work and known traps
How to monitor/resume; do not restart, stop or replace it based on stale prose.

## Unfinished publication or decisions
Actual authorization, questions and PR/check state, or explicitly none.
```

Run checks appropriate to the changes after documentation is consistent. Record
what passed, what failed and what was deferred with the reason and next check.
Do not launch a new training run merely to complete a handover.

## Publication and delivery

Commit, create/update a PR, push or merge only within the user's authorized scope
and repository rules. A local handover alone does not require publication. For an
authorized PR, prepare its concrete diff, description and validation, use exact
multiline text, and recheck checks/merge state before an authorized merge. Never
report an open PR as merged or delete a branch with unresolved work.

Use Claude Code's shell and available background-task controls for watches; retain
existing handles across timeouts and do not assume Codex-specific process tools.
Deliver the brief's path, completed changes, validation and outstanding work.
Do not pause or complete an active goal unless the user's instruction and the
goal's actual status justify it.
