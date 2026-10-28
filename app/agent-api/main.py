"""
FastAPI service running the Lar P&C risk triage pipeline
(../../eval/lar_agent/agent.py). It is an INTERNAL service: only the Express
backend calls it, and every route except /healthz requires the shared
`X-Internal-Token`, so a caller who can reach the port still can't use it or
spoof a reviewer's identity.

  POST /assess         -> intake -> redact -> injection guard -> guideline
                           retrieval -> 3-way parallel LLM fan-out ->
                           consolidation. Returns the evidence plus an
                           `evidence_signature`, WITHOUT deciding bind/refer.
                           Nothing is kept in memory: the caller holds the
                           signed evidence, so a restart loses nothing.

  POST /assess/decide   -> takes that signed evidence back with the human's
                           decision and identity. It verifies the signature
                           (evidence can't be forged or edited in between),
                           rejects an assessment_id that is already in the
                           audit log (no double decisions), runs the decision
                           through HumanJuryNode, and appends it to the
                           hash-chained audit log (audit_log.py).

  GET /ledger, /ledger/verify -> read and verify that log.
  GET /healthz (liveness), /readyz (log verifies) -> unauthenticated, minimal.

Config (environment variables):
  APP_ENV                 development (default) | production. Production
                          refuses to start with default or short secrets.
  TRIAGE_HMAC_SECRET      signing key for the audit log and evidence
  TRIAGE_INTERNAL_TOKEN   shared secret the backend sends as X-Internal-Token
  TRIAGE_AUDIT_LOG_PATH   the append-only audit log (JSONL)
  TRIAGE_MODEL            LLM for the branches (default ollama/qwen2.5:14b)
  TRIAGE_RAG              1 (default) grounds each branch in retrieved
                          underwriting guidelines; 0 disables retrieval
"""
import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
# The pipeline's prompts, helpers and policy registration live in eval/ so the
# app and the evaluation harness exercise the exact same code.
sys.path.insert(0, str(HERE.parent.parent / "eval" / "lar_agent"))

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from lar.executor import GraphExecutor
from lar.node import LLMNode, ToolNode, BatchNode, HumanJuryNode
from lar.compliance import PromptInjectionGuard, BranchTriageNode, RiskScorerNode, AuthorityLedger

from audit_log import GENESIS, AuditLog, AuditLogCorrupt, canonical
import agent as lar_agent  # reuses prompts, constants, helper functions, and
                            # the PolicyRegistry registration that already
                            # runs at import time inside agent.py

# --- configuration -----------------------------------------------------------

DEV_SECRET = "dev-only-not-a-real-secret"
DEV_TOKEN = "dev-only-internal-token"
MIN_SECRET_LEN = 32

ENV = os.environ.get("APP_ENV", "development")
HMAC_SECRET = os.environ.get("TRIAGE_HMAC_SECRET", DEV_SECRET)
INTERNAL_TOKEN = os.environ.get("TRIAGE_INTERNAL_TOKEN", DEV_TOKEN)
AUDIT_LOG_PATH = os.environ.get("TRIAGE_AUDIT_LOG_PATH", str(HERE / "data" / "audit_log.jsonl"))
RAG_ENABLED = os.environ.get("TRIAGE_RAG", "1") != "0"
MAX_SUBMISSION_CHARS = 20_000  # bounds LLM cost and abuse per request
MAX_RATIONALE_CHARS = 2_000

log = logging.getLogger("triage.agent")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def _event(**fields):
    """One JSON object per line, so logs are machine-parseable. Never put a
    submission, rationale or secret in here."""
    log.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "svc": "agent-api", **fields}))


def validate_config():
    problems = []
    for name, value, default in (
        ("TRIAGE_HMAC_SECRET", HMAC_SECRET, DEV_SECRET),
        ("TRIAGE_INTERNAL_TOKEN", INTERNAL_TOKEN, DEV_TOKEN),
    ):
        if value == default:
            problems.append(f"{name} is the public development default")
        elif len(value) < MIN_SECRET_LEN:
            problems.append(f"{name} must be at least {MIN_SECRET_LEN} characters")
    if ENV == "production":
        if problems:
            raise RuntimeError("refusing to start in production: " + "; ".join(problems))
    elif problems:
        _event(level="warn", msg="development secrets in use", problems=problems)


validate_config()

_docs = {} if ENV != "production" else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title="Risk Triage Agent API", **_docs)  # no CORS: browsers never call this service


def require_internal(x_internal_token: Optional[str] = Header(default=None)):
    if not x_internal_token or not hmac.compare_digest(x_internal_token, INTERNAL_TOKEN):
        raise HTTPException(status_code=401, detail="missing or invalid internal token")


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        if request.url.path != "/healthz":  # polled constantly by Docker; would bury real traffic
            _event(level="info", msg="request", request_id=request_id, method=request.method,
                   path=request.url.path, status=status, ms=round((time.time() - t0) * 1000))


# Durable, append-only record of every human decision (see audit_log.py).
audit_log = AuditLog(AUDIT_LOG_PATH, HMAC_SECRET)
_decide_lock = threading.Lock()  # one decision at a time


def _evidence_key() -> bytes:
    # Domain-separated from the audit-log signing key so a signature from one
    # context can never be replayed as the other.
    return hmac.new(HMAC_SECRET.encode(), b"evidence-signing-v1", hashlib.sha256).digest()


def sign_evidence(assessment_id: str, submission_sha256: str, evidence: dict) -> str:
    payload = canonical({"assessment_id": assessment_id, "submission_sha256": submission_sha256, "evidence": evidence})
    return hmac.new(_evidence_key(), payload.encode(), hashlib.sha256).hexdigest()


class AssessRequest(BaseModel):
    submission: str = Field(max_length=MAX_SUBMISSION_CHARS)


class Identity(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    role: str = Field(min_length=1, max_length=50)


class DecideRequest(BaseModel):
    assessment_id: str = Field(min_length=1, max_length=64)
    submission_sha256: str = Field(min_length=64, max_length=64)
    evidence: dict
    evidence_signature: str
    decision: str  # "bind" or "refer" -- the REAL human's choice
    rationale: str = Field(max_length=MAX_RATIONALE_CHARS)
    decided_by: Identity
    assessed_by: Optional[str] = Field(default=None, max_length=200)


def _build_phase1_graph():
    """intake -> redact -> injection guard -> fan-out -> branch triage ->
    confidence aggregation. Stops BEFORE RiskScorerNode/HumanJuryNode --
    this phase never decides bind/refer, it only produces the evidence a
    human decision needs."""
    if RAG_ENABLED:
        branch_nodes = lar_agent.build_rag_branch_nodes()
    else:
        branch_nodes = [
            LLMNode(
                model_name=lar_agent.MODEL, system_instruction=system,
                prompt_template="Submission:\n{raw_submission}", output_key=key, next_node=None,
            )
            for system, key in (
                (lar_agent.PROPERTY_SYSTEM, "property_analysis"),
                (lar_agent.FINANCIAL_SYSTEM, "financial_analysis"),
                (lar_agent.REGULATORY_SYSTEM, "regulatory_analysis"),
            )
        ]
    batch_node = BatchNode(nodes=branch_nodes, next_node=None)
    # With RAG on, retrieval runs once before the fan-out (see agent.py).
    first_stage = batch_node
    if RAG_ENABLED:
        first_stage = ToolNode(
            tool_function=lar_agent._retrieve_guidelines, input_keys=["__state__"],
            output_key=None, next_node=batch_node,
        )

    confidence_node = ToolNode(
        tool_function=lar_agent._aggregate_confidence, input_keys=["__state__"], output_key=None, next_node=None,
    )
    branch_triage = BranchTriageNode(
        branch_output_keys=["property_analysis", "financial_analysis", "regulatory_analysis"],
        risk_level_key="risk_level", finding_key="finding", critical_threshold="HIGH",
        summary_state_key="branch_findings_summary", critical_flag_key="branch_critical",
        next_node=confidence_node,
    )
    batch_node.next_node = branch_triage

    injection_guard = PromptInjectionGuard(
        input_keys=["raw_submission"], block_on_detection=True, next_node=first_stage,
    )
    redact_node = ToolNode(
        tool_function=lar_agent._redact_for_log, input_keys=["__state__"], output_key=None, next_node=injection_guard,
    )
    intake_node = ToolNode(
        tool_function=lar_agent._validate_intake, input_keys=["__state__"], output_key=None, next_node=redact_node,
    )
    return intake_node


def _classify_failure(msg: str) -> dict:
    if "Adversarial prompt injection" in msg:
        return {"status": "blocked", "reason": msg}
    if "[INTAKE_REJECTED]" in msg:
        return {"status": "rejected_at_intake", "reason": msg}
    return {"status": "error", "reason": msg}


@app.post("/assess", dependencies=[Depends(require_internal)])
def assess(req: AssessRequest):
    start_node = _build_phase1_graph()
    executor = GraphExecutor(hmac_secret=HMAC_SECRET)
    t0 = time.time()
    last = None
    try:
        for step in executor.run_step_by_step(start_node, {"raw_submission": req.submission}):
            last = step
    except Exception as e:
        return _classify_failure(f"{type(e).__name__}: {e}")
    elapsed = time.time() - t0

    # GraphExecutor reports a failing node as a final log step with
    # outcome="error" instead of raising, so the except above is not enough:
    # without this check a blocked or rejected submission fell through and
    # was presented as a normal assessment with empty findings.
    if last is None or last.get("outcome") == "error":
        return _classify_failure((last or {}).get("error") or "no steps executed")

    sa = last.get("state_after", {})
    assessment_id = uuid.uuid4().hex
    # Hash only: the submission can contain PII, and the audit log is not the
    # place to copy it. The hash still lets an auditor confirm which submission
    # a decision was about.
    submission_sha256 = hashlib.sha256(req.submission.encode()).hexdigest()
    evidence = {
        "branch_findings_summary": sa.get("branch_findings_summary"),
        "branch_critical": sa.get("branch_critical"),
        "model_confidence": sa.get("model_confidence"),
        "property_analysis": sa.get("property_analysis"),
        "financial_analysis": sa.get("financial_analysis"),
        "regulatory_analysis": sa.get("regulatory_analysis"),
        "guidelines_consulted": sa.get("guidelines_consulted"),
    }
    return {
        "status": "needs_review",
        "assessment_id": assessment_id,
        "elapsed_seconds": round(elapsed, 1),
        "submission_sha256": submission_sha256,
        # Regex-redacted (PPSN, phone, email only; names and addresses are NOT
        # caught): this is what the caller may store and show, never the raw text.
        "submission_redacted": sa.get("redacted_submission_for_log"),
        "evidence": evidence,
        "evidence_signature": sign_evidence(assessment_id, submission_sha256, evidence),
    }


@app.post("/assess/decide", dependencies=[Depends(require_internal)])
def decide(req: DecideRequest):
    if req.decision not in ("bind", "refer"):
        raise HTTPException(status_code=400, detail="decision must be 'bind' or 'refer'")
    if not req.rationale.strip():
        raise HTTPException(status_code=400, detail="rationale is required")

    expected = sign_evidence(req.assessment_id, req.submission_sha256, req.evidence)
    if not hmac.compare_digest(expected, req.evidence_signature):
        raise HTTPException(status_code=400, detail="evidence signature invalid: evidence was not produced by this service or was altered")
    evidence = req.evidence

    with _decide_lock:
        # The audit log IS the record of what has been decided, so duplicate
        # protection survives restarts and needs no separate state.
        if audit_log.has_assessment(req.assessment_id):
            raise HTTPException(status_code=409, detail="assessment already decided")

        def human_decision_provider(ctx: dict):
            # The decision was already made by a real person in the frontend
            # before this request was even sent -- this just carries it through
            # HumanJuryNode's real validation + ledger-writing path.
            return req.decision, req.rationale

        lar_ledger = AuthorityLedger(hmac_secret=HMAC_SECRET)  # per decision; durable storage is audit_log
        final_record = ToolNode(
            tool_function=lar_agent._record_final_decision, input_keys=["__state__"], output_key=None, next_node=None,
        )
        jury = HumanJuryNode(
            prompt="Underwriting decision for this submission",
            choices=["bind", "refer"],
            output_key="human_decision",
            context_keys=["branch_findings_summary", "branch_critical"],
            next_node=final_record,
            authority_ledger=lar_ledger,
            stakeholder_id=req.decided_by.id,
            stakeholder_role=req.decided_by.role,
            action_description="Bind/refer decision on P&C submission after 3-way risk fan-out",
            risk_score_key="model_confidence",
            human_decision_provider=human_decision_provider,
        )
        risk_scorer = RiskScorerNode(
            next_node=final_record, jury_node=jury,
            confidence_key="model_confidence", action_type_key="action_type",
        )

        executor = GraphExecutor(hmac_secret=HMAC_SECRET)
        initial_state = {
            "branch_findings_summary": evidence.get("branch_findings_summary"),
            "branch_critical": evidence.get("branch_critical"),
            "model_confidence": evidence.get("model_confidence"),
            "action_type": "PROPERTY_RISK_SUBMISSION",
        }
        last = None
        for step in executor.run_step_by_step(risk_scorer, initial_state):
            last = step
        sa = last.get("state_after", {})
        lar_record = lar_ledger.get_records()[-1]

        try:
            entry = audit_log.append(
                assessment_id=req.assessment_id,
                submission_sha256=req.submission_sha256,
                evidence=evidence,
                decision=req.decision,
                rationale=req.rationale,
                stakeholder_id=req.decided_by.id,
                stakeholder_role=req.decided_by.role,
                assessed_by=req.assessed_by,
                computed_oversight_level=sa.get("computed_oversight_level"),
                lar_record=lar_record,
            )
        except AuditLogCorrupt as e:
            # Fail closed: refuse to extend a chain that no longer verifies.
            raise HTTPException(status_code=503, detail=f"audit log is not intact, decision NOT recorded: {e}")

    _event(level="info", msg="decision recorded", assessment_id=req.assessment_id, seq=entry["seq"],
           decision=req.decision, decided_by=req.decided_by.id)
    return {
        "status": "decided",
        "final_decision": sa.get("final_decision"),
        "computed_oversight_level": sa.get("computed_oversight_level"),
        "audit_entry": {"seq": entry["seq"], "entry_hash": entry["entry_hash"]},
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    ok = audit_log.verify()["ok"]
    return {"ready": ok}


@app.get("/ledger", dependencies=[Depends(require_internal)])
def ledger(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    """The recorded decisions, oldest first. `head_hash` and `count` cover the
    WHOLE log (not just this page): record them somewhere independent to be
    able to detect later truncation, which verification alone cannot see."""
    try:
        entries = audit_log.entries()
    except AuditLogCorrupt as e:
        raise HTTPException(status_code=503, detail=f"audit log is not intact ({e}); see /ledger/verify")
    return {
        "count": len(entries),
        "head_hash": entries[-1]["entry_hash"] if entries else GENESIS,
        "offset": offset,
        "entries": entries[offset:offset + limit],
    }


@app.get("/ledger/verify", dependencies=[Depends(require_internal)])
def ledger_verify():
    return audit_log.verify()
