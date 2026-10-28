# Risk Triage

An underwriting triage application: a person signs in, submits a property risk,
three LLM analyses run in parallel against retrieved underwriting guidelines, and
**a human decides** bind or refer. Every decision, with the exact evidence the
reviewer saw and who they were, goes into a tamper-evident audit log. Built end
to end as a portfolio project for a generative-AI engineering role in insurance.
All submissions are synthetic; no real client data, and it is not any insurer's
system.

```
Browser ──▶ frontend (React + TS, nginx) ──▶ backend (Node + Express + TS, SQLite) ──▶ agent-api (Python, FastAPI, Lár) ──▶ model
             static, strict CSP               accounts, sessions, roles, history       internal only; signed evidence;
                                                                                         hash-chained audit log
```

## Run it

Needs Docker and [Ollama](https://ollama.com) running with the model pulled
(`ollama pull qwen2.5:14b`).

```bash
docker compose up --build        # then open http://localhost:5173
docker compose logs backend | grep -A2 "DEMO LOGINS"     # generated demo password
```

On first start the backend creates two demo accounts, `underwriter@demo.local`
and `auditor@demo.local`, with a random password printed once to its log (set
`DEMO_PASSWORD` to choose it). This is **development mode** only: production
mode refuses to seed demo users.

For a production-style deployment (required secrets, strict startup checks,
TLS terminated by your proxy) see [docs/OPERATIONS.md](docs/OPERATIONS.md).
Without Docker: `pip install -r requirements.txt`, then `uvicorn main:app --port 8010`
in `app/agent-api`, and `npm ci && npm run dev` in `app/backend` and `app/frontend`.

## What it does

| Role | Can |
|---|---|
| **Underwriter** | submit a risk, read the three findings and the guidelines each was grounded in, record Bind/Refer with a rationale, see their own history |
| **Auditor** | read everything (all history, the audit log), verify the audit chain; cannot submit or decide |

The pipeline behind a submission: intake validation → PII redaction (PPSN,
phone, email) → prompt-injection guard → guideline retrieval (RAG) → three
independent LLM analyses (property, financial, regulatory) → worst-branch
consolidation → oversight scoring → **human decision** → audit record. A
consolidated risk is the *worst* of the three, never an average.

## What makes it production-minded

- **Accounts and access control.** scrypt password hashing, HttpOnly
  `SameSite=Lax` sessions stored server-side as hashes, role checks on every
  route, login rate limiting, CSRF origin checks. The UI hides what a role can't
  do, but the server is what enforces it.
- **The browser is never trusted for evidence or identity.** It sends an
  assessment id, a decision and a rationale. The findings come from the
  server's own record, signed by the agent service; the reviewer's identity
  comes from the session. A forged or edited finding is rejected before anything
  is recorded.
- **An audit log that survives restarts and shows tampering:** hash-chained,
  HMAC-signed, append-only, verifiable by an auditor in the UI, and fail-closed
  (a decision is refused, loudly, if the log no longer verifies).
- **Operational hygiene:** validated configuration that refuses unsafe
  production settings; JSON logs with request ids across services and
  credentials redacted; liveness and readiness endpoints; graceful shutdown;
  agent-call timeouts with honest status codes (400/403/404/409/429/502/503/504);
  versioned database migrations; request size and rate limits.
- **Hardened containers:** non-root, all capabilities dropped, read-only root
  filesystems, internal-only agent service, a strict CSP with fonts bundled
  (no third-party requests), healthchecks.

[docs/SECURITY.md](docs/SECURITY.md) maps each threat to its control and the
test that covers it, **and lists what is not protected**. Read that before
treating this as more than it is.

## What "production-grade" means here, and doesn't

This is built and tested to a production standard for what it covers, and the
limits are written down rather than hidden. It has **not** been
penetration-tested, has no TLS termination of its own, no MFA or SSO, a
single-node SQLite/file design, a weak (phrase-list) prompt-injection guard, and
regex-only PII redaction. It also depends on a local model whose output varies
run to run. Those are the gaps between this and a regulated deployment; each is
listed with its mitigation in [docs/SECURITY.md](docs/SECURITY.md).

## Tests and CI

```bash
cd app/backend  && npm test        # 99 tests: auth, roles, CSRF, validation, sessions, config, logging
cd app/frontend && npm test        # 31 tests: login, role-based views, decision flow, error states
pip install -r requirements-dev.txt && pytest app/agent-api   # 71 tests, no model needed
```

The backend and agent tests stub each other, so they run without a model. The
security-critical behaviours were **mutation-checked**: each protection was
removed in turn and a test had to fail. One live test needs Ollama and is
skipped unless `TRIAGE_LIVE=1`. GitHub Actions runs typecheck, tests, lint,
builds, `npm audit` and `pip-audit`, validates both compose files (and checks
the production one refuses to start without secrets) and builds the images.
Nothing in CI calls a model, so model behaviour is covered by the evaluation
below, not by CI.

## Evaluation

`eval/` runs the same 10 labelled cases (`eval/cases.json`) through two
implementations of the identical pipeline on the same local model:
`eval/lar_agent` (Lár) and `eval/langchain_agent` (LangChain/LangGraph, with
the compliance layer hand-written). `eval/SPEC.md` is the contract both
follow.

| | Lár | LangChain/LangGraph |
|---|---|---|
| Decision accuracy | 10/10 | 10/10 |
| False negatives (REFER called BIND) | 0 | 0 |
| Severity tier exactly right | 7-8/10 across runs | 7/10 |
| Lines of code | 404 | 522 (188 of them hand-written compliance logic) |

```bash
cd eval/lar_agent && python run_eval.py            # baseline
cd eval/lar_agent && python run_eval.py --rag      # with guideline retrieval
cd eval/langchain_agent && pip install -r requirements.txt && python eval_harness.py
```

### RAG on vs off (Lár, same 10 cases, same model)

| | Baseline | With RAG |
|---|---|---|
| Decision accuracy | 10/10 (9/10 on one of four runs) | 10/10 (one run) |
| False negatives | 0 | 0 |
| Severity tier exactly right | 7-8/10 across runs | 8/10 (one run) |
| Cases where a finding cites a retrieved guideline id | n/a | 7 of 8 (the other two are the blocked and empty cases) |
| Wall clock for 10 cases | 60 s | 92 s |

Read this carefully rather than as a win: the baseline already gets every
decision right, so there is no accuracy headroom for RAG to show. What it adds
is traceability, since a finding can now point at the guideline behind it, at a
cost of about 50% more latency. The citation count is a string match on
guideline ids, so it shows the model *referenced* a guideline, not that it
applied it correctly. The guidelines corpus is synthetic and I wrote it with
this domain in mind, so the severity results are not evidence that retrieval
generalises to real guidelines. Retrieval is lexical (BM25), chosen for a
corpus this small because it is deterministic and testable; hybrid with
embeddings is the next step if the corpus grows. Per-case output:
`eval/lar_agent/eval_results.json` and `eval_results_rag.json`.

The decision metric is the one that matters here, since a missed high-risk
submission is the expensive failure. The severity misses are over- or
under-calls by one tier (HIGH vs CRITICAL) that never flipped a final
decision in these runs. That is a statement about these runs, not a
guarantee. Build notes and full output: `eval/lar_agent/NOTES.md`,
`eval/langchain_agent/NOTES.md`.

## Audit trail

Every decision is appended to `audit_log.jsonl` (a Docker volume). Each line
records the decision, the rationale, the oversight level, **who decided** (the
user id and role, set by the server from the session) and **who submitted**, a
hash of the submission (never its text), and the full evidence the reviewer was
shown. Each line is signed with HMAC-SHA256 over its content *and the previous
line's hash*, so editing, deleting or reordering an entry breaks the chain from
that point. Auditors see the log and can verify it from the UI (Audit log tab)
or the API:

```bash
GET /api/ledger?limit=20     # entries, oldest first, plus count and head_hash (auditor only)
GET /api/ledger/verify       # re-reads the file; reports the first broken entry
```

The log is read back from disk on every append, so a restart continues the
chain, and "already decided" is answered from the log itself, so double
decisions are refused across restarts too. If verification fails, new decisions
get HTTP 503 ("decision NOT recorded") instead of being appended behind the
break, and the reviewer's assessment is kept so they can retry once it's repaired.

What this does not protect against (see [docs/SECURITY.md](docs/SECURITY.md)):
- **Truncation from the end.** A shorter chain is still a valid chain. Record
  `head_hash` and the count somewhere the log's owner can't edit and compare
  later; verification alone can't see this (a test pins it).
- **Anyone with the signing secret** can rewrite and re-sign the whole file.
- **No key rotation:** entries carry no key id, so changing the secret makes old
  entries fail verification (interim procedure in the operations doc).
- **One process:** appends are serialised in-process only.

## Design decisions worth knowing

- **The agent service is stateless.** It returns evidence plus a signature; the
  backend stores both, and the agent verifies the signature when the decision
  comes back. A restart loses no pending reviews, and there is no server-side
  session state in Python to keep consistent.
- **Why three tiers.** Python is where the model work lives; the browser can't
  safely call it. The Express layer authenticates, validates, rate-limits and
  maps failures to honest status codes, which is its whole job.
- **Honest failures.** 401 and 403 are the caller's identity and rights, 404 and
  409 the caller's mistake, 429 their rate, 502/504 a dependency, 503 "the audit
  log refused the write". The UI shows the server's reason instead of a blank page.
- **Another user's assessment looks exactly like a missing one** (same 404), so
  ids can't be probed.
- **Blocked and rejected submissions are recorded,** since an injection attempt
  is a signal; they never reach the decision step.

## Known limitations

- Model output varies run to run, and the model can state facts the submission
  never gave (for example "no flood zone" when flood was not mentioned). A human
  reviews every finding; that is the design, not a guarantee of accuracy.
- The injection guard is a phrase list; a paraphrased attack is not caught.
- PII redaction is regex-only. Names and street addresses are not detected, so
  they can be stored in assessment history (only the redacted text is stored).
- Single node: SQLite, a file audit log, in-memory rate limits.
- Password-only authentication; accounts are created from the command line.
- No TLS termination, no penetration test, no compliance assessment (DPIA etc.).
- The unfinished-review state is kept in the app database, not the agent, so a
  database loss loses pending reviews (decisions are in the audit log).
- Severity-tier scores moved between 7/10 and 8/10 across identical eval runs,
  and one baseline run got `c09` wrong (a false alarm, the safe direction), so
  treat single-run eval numbers as indicative.

## Earlier version

[`marsh-risk-triage-agent`](https://github.com/axdithyaxo/marsh-risk-triage-agent) (older name, kept so existing links work)
is the first, single-agent version of this idea.
