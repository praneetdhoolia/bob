# Next agent brief

Written: 2026-10-13 19:56 +11:00. Repository state: main at 128076b (4 commits). The worktree was clean at that commit; this brief is in the commit after it. Branch: main. Remote `origin` (github.com/praneetdhoolia/bob) has **not** been pushed.

## Goal and standing instructions

Keep the full goal in [GOAL.md](GOAL.md).

**The goal:**
* Build a live-state model whose mathematical understanding of audio, video, text and world context updates within milliseconds.
* When probed, the model gets an "urge" to output a stimulus.
* A custom English interpreter of that stimulus must be verified accurate.
* Total system RAM must stay at or below 50 GB.
* Use deep research into brains and current methods, and avoid an LLM core.
* Do not dismiss the goal as impossible, but never claim consciousness from narrow tests.

**This session's instruction:**
* Run a fresh project report.
* Fix bugs and fallacies, and incorporate the research.
* Bring every report area to at least 4/5.
* Verify with a second report, then hand off.
* The focus is organizing and submitting the initial version, not achieving the goal.

**Constraints:** see the [project map](agent-workflows.md):
* port 3217 only;
* the 50,000,000,000-byte RAM guard;
* large offline jobs share `.runtime/research.lock`;
* candidate checkpoints stay isolated from the live selection;
* no private camera or audio data in reports.

## Verify first

| Fact observed at handoff | Evidence/handle | Recheck |
|---|---|---|
| Worktree clean at 128076b; no push | `git status`, `git log` | `git status --short; git log --oneline -5; git status -sb` |
| Live agent not running; no listener on 3217; no Python processes | — | `npm run status` |
| Full suite passes: 40 Python, 10 Node, evidence manifest, core invariants (6/10, quiet 120/120) | npm test on 13c7151 | `npm test` |
| Selected live weights: v1 checkpoint migrated into v7 (4,663,062 parameters; 610,567 inert) | `artifacts/liquid-core-training.json` | Read it; `scripts/preflight.py` |
| The user's saved live state still uses the old single-file format with embedded weights. It loads, and the first save writes `liquid-live-state.weights.pt` | `artifacts/liquid-live-state.pt` (18.7 MB) | After first start, check that both files exist |

## What changed and what was measured

Reports:
* [Report 3](reports/20261013T081400_project_report.html) is the assessment.
* [Report 4](reports/20261013T085200_project_report.html) is the verification.
* The ledger is [recommendations.json](reports/recommendations.json): R18–R28 were added in report 3 and R29–R34 are open.
* Corrections and new results are at the top of [live-core-results.md](live-core-results.md).

**Fallacies found and fixed or disclosed**
* Validation video is bit-identical to training. The old "6/10 held-out" was 5 familiar-frame video answers, 1 control and 0/4 unseen words.
* Scoring was lenient; it is now `match_targets`, strict.
* A start-node baseline (167/512) beat the four-hop results; `hop_baselines` now reports it.
* Persistence was scored on mismatched packets.
* Silence states included answer windows; quiet is now 120/120.

**New mechanisms**
* `neural/urge.py`: an urge accumulator gates onset and releases a `stimulus` event.
* `neural/evaluate_interpreter.py`: the verification protocol.
* `--residual` forecast mode in `probe_continual_context.py`.
* `neural/probe_runtime_live.py`: an isolated active runtime.

**Measured**

Fruit interpreter (receipt `docs/evidence/receipts/interpreter-verification-1791881348613247400.json`):
* depends on the held state: null hit 8/168, identification 0.94;
* unseen words 0/4, ECE 0.27.

Video interpreter:
* follows current frames: intervention flip 1.0;
* familiar frames only.

Urge on complete streams:
* gated 11/28, ungated 12/28;
* clamping surprise delays onset from 50 to 73 ms.

Residual forecast:
* At lr 1e-6, held skill is +0.030; shuffling the state gives −0.0005.
* At 1e-4 it diverges.
* One seed.

Isolated live run:
* 74 ticks/s against a nominal 100, and 9.95 observations/s.
* One stimulus release.
* Malformed utterance "s bidinnPear."
* Exact restore.
* Receipt: `artifacts/runtime-live-1791880742764717500/validation.json`.

**Runtime, server and UI repairs** are listed in report 4 and in the commit messages.

**Evidence:** cited receipts are copied byte-exactly under `docs/evidence/receipts/` and verified by `scripts/snapshot_evidence.py --check`. Rerun `scripts/snapshot_evidence.py` after citing any new receipt, or `npm test` fails.

## Next work

Priority order. Each has its success check in the ledger.
1. **R29 held-out media.** Encode held frames, held scenes and new voices. Today no claim of visual generalization is possible. Success: interpreter verdicts on unseen visual input.
2. **R30 forecast.** Run the residual forecast with 3 seeds and horizons from 100 ms to 2 s, with bootstrap intervals. Success: skill ≥ 0.05 at ≥ 500 ms. Start from `--residual --learning-rate 1e-6`.
3. **R32 interpreter training.** Train for unseen words and calibrated confidence on the same state. Rerun `evaluate_interpreter.py` and compare its verdicts.
4. **R31 lock contention.** Success: the live probe reaches at least 95 ticks/s.
5. **R33 soak test.** Run a soak and hardware-capture test only with the user present: camera and microphone are private.
6. **Later indicator assays.** Workspace competition, attention schema and interoceptive regulation; see the report 3 research section.

## Running work and known traps

* **Running work:** none.
* **Live probes:**
  * `probe_runtime_live.py` uses the GPU encoders and the research lock. It opens no port, but it does read the built-in recordings.
  * Do not run it alongside training; jobs queue on the lock.
* **Evaluation protocol:**
  * `evaluate_interpreter.py` assumes the selected checkpoint's 0.15 s training protocol.
  * Pass `--input-period .1` for newer `train_liquid_core.py` checkpoints.
* **Learning rate:** Adam at lr ≥ 1e-5 destabilizes the residual forecast, because the 2,048-wide normalized readout amplifies coherent per-weight steps.
* **Line endings:** `core.autocrlf=true`. Evidence receipts are marked `-text` in `.gitattributes`; keep it that way or the hashes break on checkout.
* **Retired folders:** about 47 GB in `.runtime` (see USAGE). Deleting them needs user approval; the earlier automatic deletion was blocked by policy.

## Unfinished publication or decisions

* **Commits:** the user asked to organize and submit the initial version. Local commits were made. Pushing to `origin` has **not** been done, and it needs the user's explicit go-ahead, because it publishes the research PDF, receipts and code.
* **Retired folders:** deleting them (R34) awaits the user's decision.
* **Pull requests and CI:** none exist.
