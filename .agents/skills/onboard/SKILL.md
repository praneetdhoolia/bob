---
name: onboard
description: Reconstruct a repository's goal, current implementation, measured results and next work from evidence. Use for session onboarding, a catch-up, project status or $onboard.
---

# Onboard

Assume no session memory. Deliver a concise, evidence-backed briefing that another
agent can use to resume the user's actual task. This adapts the source workflow's
four questions and drift scan without assuming its simulator or document system.

## Reconstruct

1. Locate the repository root and applicable instructions. Read the entrypoint,
   usage guide, goal/design document, current results and latest handover if present.
   Search headings and open only the sections needed; a reading budget must not
   hide a requirement or an unresolved failure.
2. For Bob, use the [project map](../../../docs/agent-workflows.md). For another
   repository, discover its equivalent files and supported commands rather than
   inventing a status board, session gate or decision ledger.
3. Inspect the worktree, branch and available commit history. Treat uncommitted
   and untracked files as in-flight work. An unborn branch has no verified HEAD.
   Read remote issues/PRs when relevant and access is available; mark missing
   remote evidence rather than blocking local onboarding.
4. Recheck expiring facts: listeners, process identity, job progress, selected
   checkpoint, resource use, open checks and PR state. A PID/receipt alone does
   not prove a job is alive. An observation timeout does not prove it stopped.
5. Run only appropriate lightweight checks. During active computation, defer
   resource-heavy validation through the project's scheduling mechanism. Report
   a failed check and its scope; do not claim that discovery or syntax proves behavior.

## Answer the four questions

- **Distance to the goal:** each requested capability is verified, incomplete,
  contradicted or unverified, with its evidence and the scope of that evidence.
- **Current phase and next work:** implementation/experiment stage, the next
  concrete task, its expected cost and any actual dependency.
- **Measured quality:** latest completed results and live limitations; distinguish
  training from held-out results, completed from partial runs, and comparable
  controls from measurements with changed conditions.
- **Unfinished business:** active jobs and their handles, in-flight changes,
  failing checks, decisions, publication state and unresolved findings.

Scan for drift between documents, code, receipts and live behavior. Present gaps
without silently rewriting evidence or fixing unrelated work during onboarding.
State standing user decisions accurately; do not treat approvals as expired merely
because the session changed. Ask only for information that actually blocks the task.

Use Codex process sessions for authorized watches, preserving their handles and
using bounded waits. Do not assume Claude Monitor, slash-command tools or hosted
Artifacts. Finish after the briefing when onboarding is the whole request;
otherwise continue work already authorized by the user.
