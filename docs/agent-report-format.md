# Project report format and ledgers

This is the shared report contract for the ChatGPT/Codex and Claude Code skills.
The report directory defaults to `docs/reports/`; use the project's existing
equivalent when adapting the workflow elsewhere. Scratch inputs are disposable,
and dated report files are never overwritten.

## Assessment JSON

The renderer consumes this structure. Text is escaped, not interpreted as HTML
or Markdown. Links use explicit `label` and HTTP(S) `url` objects. Local evidence
can be a `file:line` or artifact path alongside the claim.

```json
{
  "title": "Project assessment",
  "generatedAt": "ISO-8601 timestamp with timezone",
  "summary": "Concrete verdict and its most important limitation.",
  "sections": [
    {
      "heading": "Goal coverage",
      "paragraphs": ["Scope and evidence reviewed."],
      "bullets": ["A limitation or next action."],
      "table": {
        "columns": ["Requirement", "Status", "Evidence", "Next measurement"],
        "rows": [["Requested capability", "Unverified", "Source or artifact", "Verification"]]
      }
    }
  ],
  "sources": [{"label": "Primary source title", "url": "https://example.org/paper"}]
}
```

Sections should cover method/read coverage, anatomy, implementation, findings,
optimization, performance, testing, history/issues, goal coverage, research,
documentation/process, in-flight work, changes since the last report,
recommendations and limitations. Combine short sections when it improves reading.
Unavailable evidence is a stated gap; collection is not an exhaustive review.

## Durable records

- `README.md`: newest-first dated report links, inspected revision/worktree state
  and a short verdict. If there is no HEAD, say so instead of inventing a revision.
- `recommendations.json`: rows with `id`, `category`, `change`, `evidence`,
  `validation`, `status` and `statusEvidence`. Reuse IDs for continuing findings;
  preserve completed/declined status until evidence supports a change.
- `reference/field-survey.json`: comparable projects/methods with purpose, scope,
  published evaluation, strengths, limits, source URLs, publication dates and
  `lastVerified`. Record checked exclusions and unanswered questions separately.
- `reference/factors.json`: requirements/mechanisms and what primary research
  establishes, with source URLs, dates, comparison limits and research gaps.
  Re-derive each factor's status in this repository from current code/results in
  every assessment; do not carry its old implementation status as a literature fact.

Use the previous reference records before searching. Refresh claims according to
their stability and the question being answered; no universal freshness interval
turns an unverified fact into a current one. Persist useful new research and gaps
before ending a report pass. If a research phase is omitted, disclose the omission
and retain the prior library with its original dates.

In Bob, distinguish a proposed consciousness indicator, a numerically implemented
mechanism, a narrow behavioral result and an unverified subjective claim. Compare
training/held-out protocols, sensory inputs, elapsed intervals, interventions and
ablations explicitly. A paper's result is not Bob's result.
