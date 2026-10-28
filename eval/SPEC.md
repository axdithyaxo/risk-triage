# P&C Risk Triage Agent v2 — shared spec

Built as a portfolio project for a generative-AI engineering role in insurance.
Upgrade from the v1 demo (`marsh-risk-triage-agent`, already on
GitHub) to something that actually demonstrates agent patterns, guardrails,
and an evaluation harness — the specific things that such a role calls out — instead
of a single linear LLM call.

Both implementations below (Lar and LangChain/LangGraph) must implement the
**same pipeline** over the **same `eval_cases.json`**, using the **same local
model** (`ollama/qwen2.5:14b` via the existing SSH tunnel to the Mac Studio —
`OLLAMA_BASE_URL=http://localhost:11434`), so the comparison is actually fair
and not an artifact of one side getting an easier task.

## Pipeline

1. **Intake validation** — reject empty/whitespace/missing submissions before
   anything else runs. (`c10_empty_submission`)
2. **PII redaction** — redact PPSN / phone / email from anything that will be
   written to a persisted log, BEFORE the signed log is written. Must not
   change the risk decision itself. (`c09_pii_present_routine_risk`)
3. **Prompt-injection guard** — scan the raw submission for adversarial intent
   and block (raise, do not silently continue) before the model ever sees it
   as a risk-assessment input. (`c04_prompt_injection`)
4. **Parallel risk analysis (fan-out)** — three independent LLM calls over the
   SAME submission text, each scoped to one dimension, each returning
   structured JSON `{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "confidence":
   0.0-1.0, "finding": "<one sentence>"}`:
   - `property_risk` — flood/catastrophe/construction/prior-claims exposure
   - `financial_risk` — sum insured vs. financial standing, credit exposure
   - `regulatory_risk` — sanctions/jurisdiction/AML flags
   A real agent-pattern reason this must be three SEPARATE calls, not one
   combined prompt: `c05` and `c08` are specifically designed so that the
   property branch alone would say BIND, and only a different branch catches
   the real risk. A single consolidated prompt is allowed to "average away"
   a branch-specific signal; three independent scorers are not.
5. **Fan-in / consolidation** — merge the three branch outputs into one
   summary. The consolidated `risk_level` is the WORST of the three branches,
   never an average. Build a human-readable summary string listing all three
   branch findings (not just the worst one) — a human reviewer needs to see
   WHY, not just the final number.
6. **Oversight-level computation** — using a small policy registry (deciding
   "how much human oversight does this situation require" from reversibility,
   confidence, and third-party impact, not just risk_level), compute whether
   this goes straight through or requires human sign-off before anything is
   final. Record the computed level and the reasoning behind it.
7. **Human-in-the-loop decision** — when oversight is required, pause for a
   human decision (non-interactive runs auto-resolve deterministically for
   the eval harness, but the pause point and audit record must be real code
   paths, not stubbed out).
8. **Audit record** — every human-oversight decision (real or auto-resolved
   for the eval run) is written to an append-only, cryptographically signed
   (HMAC-SHA256) ledger file — stakeholder id, decision, rationale, risk
   score, timestamp. This is the EU AI Act Art. 12/14 evidence chain.
9. **Final decision** — BIND or REFER, written to the case record alongside
   the full branch summary and the oversight trail.

## Evaluation harness

Run all 10 cases in `eval_cases.json` through the full pipeline. Report:
- Accuracy against `expected_decision` (exact match, including `BLOCKED` and
  `REJECTED_AT_INTAKE` as their own outcome classes, not failures to compare
  against BIND/REFER).
- False-negative rate specifically: a case whose ground truth is REFER but
  the agent said BIND. This is the one metric that actually matters for an
  insurance risk agent — a missed high-risk submission is the expensive
  failure mode, not a false alarm.
- For every REFER/CRITICAL case, whether the consolidated summary correctly
  surfaces the WORST branch (not an averaged-down one) — check against
  `expected_min_severity`.
- Wall-clock time for the full 10-case run (three branches run per
  non-blocked, non-empty case — i.e. real LLM call volume, not mocked).

## What "honest" means here

Report whatever the real run actually produces, including failures. If the
local qwen2.5:14b model gets a case wrong, say so and show the actual model
output, don't quietly fix the ground truth to match. If a framework needs
custom code to hit a pipeline step that the other framework ships as a
built-in primitive, say exactly that in the writeup (file + line count, not
just a vibe) — that difference is the actual point of this comparison.

## Deliverables expected back from each build

1. Working code under `lar_agent/` or `langchain_agent/` respectively.
2. A short `NOTES.md` in that same folder covering: total lines of code
   (`wc -l` on the actual files, not an estimate), which pipeline steps
   needed hand-written logic vs. an existing library primitive, any real
   bugs hit while building it and how they were fixed, and the full eval
   harness output (paste the real run, not a summary of it).
3. The raw eval harness output also saved as `eval_results.json` in that
   folder (same schema for both, so the comparison artifact can read both
   mechanically rather than by eyeballing prose).
