---
name: project-report
description: Assess a whole repository's implementation, results, performance, history and research context. Produce a dated HTML report with evidence and ranked recommendations. Use for /project-report or a full project assessment.
---

# Project report

Produce a dated, self-contained HTML assessment under `docs/reports/`, or the
repository's existing report directory. Preserve the source workflow's code,
performance, history, research and recommendation audit; derive domain-specific
requirements from this project rather than carrying over another project's targets.

Read the [shared report format and ledger contract](../../../docs/agent-report-format.md)
when preparing report inputs or updating the reference/recommendation records.

The report is an assessment. Write report inputs, the dated report, its index,
recommendation ledger and research reference library. Do not modify implementation,
datasets, live weights, operational controls or issues just to make the report pass.
Fixes or publication require their own authorized scope.

## Ground and collect

Read the [project map](../../../docs/agent-workflows.md) for Bob's evidence and
runtime constraints. In another repository, discover the equivalents. Establish
the date, worktree, branch and HEAD (or absence of commits), current goal, previous
report, recommendations, reference library and active work. Recheck volatile facts.

From the repository root, use its Python interpreter:

```text
python scripts/project_report.py collect <scratch>/metrics.json --root .
```

For Bob, the project map gives its interpreter and optional local status URL.
The shared helper inventories tracked and untracked source files, excludes
generated stores, and records Git availability. It is a collector, not a reviewer.
Supplement it with code reads, relevant receipts and available read-only issue,
PR and CI history. State unavailable history and unread coverage explicitly.

## Assess

1. **Implementation:** group by actual responsibility; inspect code, concurrency,
   error paths, dependencies, data provenance, tests and accessibility. Findings
   need `file:line`, a failure scenario, severity and a reproducible basis.
2. **Optimization:** concrete redundancy, quality, efficiency and simplification
   changes; measured cost or explicitly reasoned estimate, expected saving range,
   effect on results/compatibility and the check needed. Preserve required behavior.
3. **Performance/results:** actual latency, throughput, RAM/VRAM, compute, I/O,
   training/evaluation cost and completion status where measured. Keep protocols,
   dataset splits and baselines separate. A partial run is a partial reading.
4. **History/process:** milestones, commits, available PRs/issues/CI, documentation
   drift, in-flight work and whether prior recommendations improved the goal.
5. **Goal coverage:** one evidence-backed row per real requirement or capability,
   with data it has/lacks, controls, current status and the next measurement.
6. **Research context:** read the prior reference library first. Verify current
   project status from code/artifacts every pass. Refresh external facts according
   to their stability; use primary sources, dates and explicit comparison limits.
   Record unanswered questions instead of exhausting searches on repeated variants.

Do not infer a consciousness score, general capability or zero bias from a toy
test. Do not conflate observed values, assumptions and literature estimates.
Ratings are optional; each used rating needs its rubric and supporting evidence.

## Synthesize and deliver

Deduplicate findings and rank recommendations by impact, cost and testability.
Give each recommendation an ID, category, proposed change, evidence and validation;
preserve prior ledger statuses unless fresh evidence changes them. Explain what
changed since the previous report and what the assessment did not inspect.

Create JSON with `title`, `generatedAt`, `summary`, `sections` and `sources`.
Sections accept `heading`, `paragraphs`, `bullets` and `table` (`columns`, `rows`).
Include findings, goal coverage, optimization, limitations and recommendations in
those sections. Keep citations next to their claims and embed the source JSON.

```text
python scripts/project_report.py render <scratch>/report.json docs/reports/<timestamp>_project_report.html
```

The shared renderer escapes content, embeds CSS and data, supports light/dark
appearance and scrollable tables, and refuses to overwrite an earlier report.
Inspect the result; validate cited paths, rendered content, mobile layout and
available contrast. Update the report index with a short verdict and artifact link.
Deliver that link and the highest-priority recommendations.

Work sequentially unless delegation is authorized and useful. If delegated, assign
disjoint outputs, respect actual slots, and retain one owner per area. Use native
Claude Code tools and bounded process waits; do not assume Codex process-session
tools, a particular Monitor expiry or hosted Artifacts. Publication and follow-on implementation follow existing authorization.
