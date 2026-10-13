# Agent workflows

These workflows adapt `onboard`, `handoff` and `project-report` from the sibling
repository. Their procedures are project agnostic; this page supplies Bob's map.
The source repository and its framework settings/hooks are left unchanged.

## Invocation and structure

| Framework | Repository skill path | Commands |
|---|---|---|
| ChatGPT/Codex | `.agents/skills/<name>/SKILL.md`, with `agents/openai.yaml` | `$onboard`, `$handoff`, `$project-report` |
| Claude Code | `.claude/skills/<name>/SKILL.md` | `/onboard`, `/handoff`, `/project-report` |

Both sets use the same local workflow and reporting helper. Their final tool
instructions use the corresponding framework. The OpenAI metadata keeps automatic
selection enabled; Claude uses its default discovery. No global skill installation,
API key, custom hook or unrelated repository configuration is needed. Start a new
agent session if the current session has not discovered the new skills.

The layouts follow [official OpenAI skill documentation](https://learn.chatgpt.com/docs/build-skills)
and [Claude Code skill documentation](https://code.claude.com/docs/en/skills).

## Evidence map

| Question | Current source |
|---|---|
| Intended goal and architecture | User instructions, `docs/GOAL.md`, `sentience-research-64gb.pdf`, `docs/live-core-design.md` |
| Startup, ports, controls, training | `USAGE.md`, `package.json`, `scripts/*-workspace.ps1` |
| Measured capabilities and failures | `docs/live-core-results.md` and the exact receipts it cites |
| Interface behavior and checks | `docs/interface.md`, `public/agent.*`, cited UI receipts |
| Neural implementation | `neural/liquid_core.py`, `neural/liquid_runtime.py`, sensory/training/probe modules |
| Live process and selected weights | Revalidated `.runtime/agent-server.json`, `/api/status`, `artifacts/liquid-core-training.json` |
| Candidate progress | Exact `artifacts/<experiment>/training.json`, checkpoint and a verified process/session handle |
| Handover | `docs/NEXT_AGENT_BRIEF.md` when a handoff has actually been requested and written |

Live behavior, code and receipts establish implementation/results. Design text and
the PDF establish requirements or hypotheses. A handover is a pointer, not proof.
Recheck listeners, jobs, memory, selected weights and Git/PR state each session;
do not embed today's PID, progress or scores here as permanent facts.

## Runtime and research constraints

- The live interface is `http://127.0.0.1:3217/`. Use `npm run status` and a read-only
  status request before concluding it is running. The descriptor can be stale.
- Keep authorized camera/audio, numerical state and running jobs intact. Reporting
  and onboarding do not authorize restart, source switching or checkpoint selection.
- Total system RAM has a 50,000,000,000-byte ceiling. Physical RAM capacity is a
  different value. GPU allocations and disk files are separate measurements.
- Large offline workloads use `neural/research_resources.py`'s research slot and
  prospective RAM headroom. Do not allocate a second full training/gradient workload
  to validate the first. Re-poll the existing handle after a timeout.
- Candidate checkpoints remain isolated until relevant transfer and regression
  checks pass. One common numerical state and custom readout remain the target;
  a second conversational or motor expert is not a substitute.
- Narrow recall, captions, changing numerical activity and biological analogies do
  not establish consciousness, broad English, subjective time or universal lack of bias.
- Private camera/audio, messages, secrets and full live numerical dumps do not belong
  in a shareable project report. Use synthetic fixtures for media/UI verification.

## Reporting helper

Run from the repository root, using the installed interpreter:

```powershell
& .runtime/omni-python/Scripts/python.exe scripts/project_report.py collect .agents/.local/project-report/metrics.json --root . --status-url http://127.0.0.1:3217/api/status
& .runtime/omni-python/Scripts/python.exe scripts/project_report.py render .agents/.local/project-report/report.json docs/reports/<timestamp>_project_report.html
```

Claude may use `.claude/.local/project-report/` for scratch inputs. The collector
reads source metadata and optional filtered status; it does not load neural weights,
compile, train, start/stop services or change Git. Missing Git history and unavailable
status are recorded as gaps. The renderer accepts human-authored assessment JSON;
collection alone never counts as a code review. A report request owns its dated HTML,
index, recommendations and reference library, not repairs to the assessed project.
