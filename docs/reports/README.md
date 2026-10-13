# Project reports

| Report | Inspected state | Verdict |
|---|---|---|
| [3 — Evaluation fallacies and initial-version readiness](20261013T081400_project_report.html) | main, no commits; working tree after stale-bytecode cleanup and evidence snapshot script | All areas below 4/5. Validation video is training-identical (6/10 "held-out" = 5 familiar-frame video answers + 1 control; unseen words 0/4); lenient scoring; start-node baseline beats four-hop; autosave failure stops the agent; no urge or verified interpreter. |
| [2 — Repairs and continual-learning verification](20261013T044300_project_report.html) | main, no commits; implementation hashes in assessment scratch | Engineering and research-method areas reach 4/5; functional architecture and reproducibility remain below 4. Handoff deferred. |
| [1 — Baseline](20261013T022617_project_report.html) | main, no commits; working-tree assessment | Below 4/5; runtime defects and incomplete self/world learning. |

Post-report verification is recorded in [current results](../live-core-results.md):
startup without optional demonstration media passes, mandatory sensory assets
match a pinned upstream revision, and matched temporal-credit controls isolate a
narrow forecast improvement from learning write strengths. Functional architecture
and cold-install reproduction still need work; no handoff has been performed.
