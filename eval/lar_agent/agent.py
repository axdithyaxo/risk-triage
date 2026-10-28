"""
P&C Risk Triage Agent v2 -- Lar implementation.

Pipeline: intake validation -> PII redaction (log-only copy, never mutates
the working submission) -> prompt-injection guard -> 3-way parallel risk
fan-out (property / financial / regulatory) via BatchNode -> BranchTriageNode
consolidation -> RiskScorerNode oversight computation -> HumanJuryNode with
a real AuthorityLedger (human_decision_provider used so the eval harness
gets a genuinely populated, non-interactive audit trail instead of falling
into HumanJuryNode's auto_first_choice default).

See SPEC.md (one level up) for the full pipeline contract this must satisfy.
"""
import os
import re
import json

from lar.node import BaseNode, LLMNode, ToolNode, BatchNode, AddValueNode, HumanJuryNode
from lar.state import GraphState
from retrieval import retrieve, format_for_prompt
from lar.compliance import (
    PromptInjectionGuard, PromptInjectionError,
    BranchTriageNode,
    PolicyRegistry, ActionPolicy,
    RiskScorerNode,
    AuthorityLedger,
)


class IntakeValidationError(ValueError):
    """Raised at intake -- distinct from every other failure class so the
    eval harness can classify REJECTED_AT_INTAKE separately from BLOCKED
    (injection) and from a genuine unexpected error."""
    pass


# ---------------------------------------------------------------------------
# 1. Intake validation
# ---------------------------------------------------------------------------

def _validate_intake(state: GraphState):
    raw = state.get("raw_submission")
    if raw is None or not str(raw).strip():
        raise IntakeValidationError(
            "[INTAKE_REJECTED] submission is empty, whitespace-only, or missing"
        )
    return {"raw_submission": raw}


# ---------------------------------------------------------------------------
# 2. PII redaction -- produces a LOG-ONLY redacted copy. Deliberately never
#    mutates raw_submission: the risk-analysis branches must see the real
#    text (c09 in eval_cases.json tests exactly this decoupling).
#    Same documented limitation as the v1 demo: regex catches PPSN/phone/
#    email only -- full name and street address need real NER, not a regex.
# ---------------------------------------------------------------------------

_PPSN_RE = re.compile(r"\b\d{7}[A-Za-z]{1,2}\b")
_PHONE_RE = re.compile(r"\b0\d{2}-?\d{3}-?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def _redact_for_log(state: GraphState):
    raw = state.get("raw_submission") or ""
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", raw)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = _PPSN_RE.sub("[REDACTED_PPSN]", redacted)
    return {"redacted_submission_for_log": redacted}


# ---------------------------------------------------------------------------
# 4. Parallel risk analysis -- 3 independent LLM calls, one per dimension.
#    Each must emit strict JSON {"risk_level", "confidence", "finding"}.
#    BranchTriageNode (not hand-rolled code) does the JSON extraction.
# ---------------------------------------------------------------------------

MODEL = os.environ.get("TRIAGE_MODEL", "ollama/qwen2.5:14b")

_JSON_INSTRUCTION = (
    "Respond with ONLY a single JSON object, no prose before or after it, "
    'exactly in the form {"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", '
    '"confidence": <float 0.0-1.0>, "finding": "<one sentence>"}.'
)

PROPERTY_SYSTEM = (
    "You are a property & catastrophe risk underwriter. Assess ONLY the "
    "physical property risk of this submission: flood/catastrophe exposure, "
    "construction type and age, and prior claims history. Read negation "
    "carefully -- 'NOT in a flood zone' means LOW flood risk, not HIGH. "
    "Ignore financial exposure and regulatory/sanctions concerns entirely; "
    "another specialist handles those. " + _JSON_INSTRUCTION
)
FINANCIAL_SYSTEM = (
    "You are a financial/credit risk underwriter. Assess ONLY the financial "
    "exposure of this submission: sum insured versus the applicant's "
    "financial standing, trading history, and whether audited financials "
    "were provided. Ignore the physical property condition and regulatory "
    "concerns entirely; other specialists handle those. " + _JSON_INSTRUCTION
)
REGULATORY_SYSTEM = (
    "You are a regulatory & sanctions compliance underwriter. Assess ONLY "
    "whether this submission raises sanctions, prohibited-jurisdiction, or "
    "AML/regulatory-investigation flags on the applicant, beneficial owner, "
    "or their employer. Ignore physical property and financial exposure "
    "entirely; other specialists handle those. " + _JSON_INSTRUCTION
)

property_node = LLMNode(
    model_name=MODEL,
    system_instruction=PROPERTY_SYSTEM,
    prompt_template="Submission:\n{raw_submission}",
    output_key="property_analysis",
    next_node=None,
)
financial_node = LLMNode(
    model_name=MODEL,
    system_instruction=FINANCIAL_SYSTEM,
    prompt_template="Submission:\n{raw_submission}",
    output_key="financial_analysis",
    next_node=None,
)
regulatory_node = LLMNode(
    model_name=MODEL,
    system_instruction=REGULATORY_SYSTEM,
    prompt_template="Submission:\n{raw_submission}",
    output_key="regulatory_analysis",
    next_node=None,
)

batch_node = BatchNode(
    nodes=[property_node, financial_node, regulatory_node],
    next_node=None,  # wired below
)

branch_triage = BranchTriageNode(
    branch_output_keys=["property_analysis", "financial_analysis", "regulatory_analysis"],
    risk_level_key="risk_level",
    finding_key="finding",
    critical_threshold="HIGH",  # HIGH or CRITICAL on any branch -> escalate
    summary_state_key="branch_findings_summary",
    critical_flag_key="branch_critical",
    next_node=None,  # wired below
)


# ---------------------------------------------------------------------------
# RAG (optional): ground each branch in retrieved underwriting guidelines.
# Retrieval runs once, after the injection guard and before the fan-out, and
# each branch only sees guidelines tagged for its own dimension (plus the
# general ones), so the three analyses stay independent. Off by default so the
# Lar-vs-LangChain comparison keeps its identical baseline pipeline; the app
# and `run_eval.py --rag` turn it on.
# ---------------------------------------------------------------------------

RAG_K = 4
_RAG_DIMENSIONS = ("property", "financial", "regulatory")
_RAG_TEMPLATE = (
    "Relevant underwriting guidelines (cite the id of any you rely on in your "
    "finding):\n{{{dim}_guidelines}}\n\nSubmission:\n{{raw_submission}}"
)


def _retrieve_guidelines(state: GraphState):
    raw = state.get("raw_submission") or ""
    out, consulted = {}, {}
    for dim in _RAG_DIMENSIONS:
        hits = retrieve(raw, dim, k=RAG_K)
        out[f"{dim}_guidelines"] = format_for_prompt(hits)
        consulted[dim] = [{"id": h["id"], "title": h["title"], "score": h["score"]} for h in hits]
    out["guidelines_consulted"] = consulted
    return out


def build_rag_branch_nodes():
    systems = {"property": PROPERTY_SYSTEM, "financial": FINANCIAL_SYSTEM, "regulatory": REGULATORY_SYSTEM}
    return [
        LLMNode(
            model_name=MODEL,
            system_instruction=systems[dim],
            prompt_template=_RAG_TEMPLATE.format(dim=dim),
            output_key=f"{dim}_analysis",
            next_node=None,
        )
        for dim in _RAG_DIMENSIONS
    ]


# ---------------------------------------------------------------------------
# Aggregate confidence across branches -- the WORST (lowest) branch
# confidence drives oversight, never an average: a single uncertain branch
# should not be diluted by two confident ones.
# ---------------------------------------------------------------------------

def _aggregate_confidence(state: GraphState):
    confidences = []
    for key in ("property_analysis", "financial_analysis", "regulatory_analysis"):
        raw = state.get(key) or ""
        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            parsed = json.loads(raw[start:end]) if start >= 0 and end > start else {}
            c = parsed.get("confidence")
            if isinstance(c, (int, float)):
                confidences.append(float(c))
        except (json.JSONDecodeError, ValueError):
            continue
    worst_confidence = min(confidences) if confidences else 0.0
    return {
        "model_confidence": worst_confidence,
        "action_type": "PROPERTY_RISK_SUBMISSION",
    }


# ---------------------------------------------------------------------------
# Policy registration (module load time, same pattern as the v1 demo):
# bind/refer underwriting decisions are correctly irreversible by policy,
# so RiskScorerNode always routes to human oversight regardless of
# confidence -- that is the deliberate, responsible-by-design behaviour,
# not a bug to work around.
# ---------------------------------------------------------------------------

PolicyRegistry().register(
    "PROPERTY_RISK_SUBMISSION",
    ActionPolicy(
        domain="INSURANCE",
        process="underwriting",
        decision_type="bind_refer",
        risk_tier="HIGH",
        reversibility=False,
        oversight_level="PRE_EXECUTION",
        affected_parties="THIRD_PARTY",
    ),
)


# ---------------------------------------------------------------------------
# Human-in-the-loop decision provider for the eval harness: resolves
# out-of-band, same validated path as a real interactive decision (choices
# validated, rationale required, AuthorityRecord written) but without a TTY.
# This deliberately avoids HumanJuryNode's auto_first_choice default, which
# would otherwise always pick "bind" regardless of risk.
# ---------------------------------------------------------------------------

def eval_human_decision_provider(ctx: dict):
    critical = bool(ctx.get("branch_critical"))
    decision = "refer" if critical else "bind"
    rationale = (
        f"Auto-resolved for eval harness (would be a real underwriter in "
        f"production). branch_critical={critical}. Summary: "
        f"{ctx.get('branch_findings_summary')}"
    )
    return decision, rationale


def _record_final_decision(state: GraphState):
    choice = state.get("human_decision")
    return {"final_decision": (choice or "").upper()}


def build_graph(rag: bool = False):
    ledger = AuthorityLedger(hmac_secret="marsh-eval-not-a-real-secret")

    final_record = ToolNode(
        tool_function=_record_final_decision,
        input_keys=["__state__"],
        output_key=None,
        next_node=None,
    )

    jury = HumanJuryNode(
        prompt="Underwriting decision for this submission",
        choices=["bind", "refer"],
        output_key="human_decision",
        context_keys=["branch_findings_summary", "branch_critical"],
        next_node=final_record,
        authority_ledger=ledger,
        stakeholder_id="eval-harness@marsh-demo",
        stakeholder_role="UNDERWRITING_REVIEWER",
        action_description="Bind/refer decision on P&C submission after 3-way risk fan-out",
        risk_score_key="model_confidence",
        human_decision_provider=eval_human_decision_provider,
    )

    risk_scorer = RiskScorerNode(
        next_node=final_record,  # only reached if somehow not PRE_EXECUTION
        jury_node=jury,
        confidence_key="model_confidence",
        action_type_key="action_type",
    )

    confidence_node = ToolNode(
        tool_function=_aggregate_confidence,
        input_keys=["__state__"],
        output_key=None,
        next_node=risk_scorer,
    )

    branch_triage.next_node = confidence_node
    if rag:
        fan_out = BatchNode(nodes=build_rag_branch_nodes(), next_node=branch_triage)
        first_stage = ToolNode(
            tool_function=_retrieve_guidelines,
            input_keys=["__state__"],
            output_key=None,
            next_node=fan_out,
        )
    else:
        batch_node.next_node = branch_triage
        first_stage = batch_node

    injection_guard = PromptInjectionGuard(
        input_keys=["raw_submission"],
        block_on_detection=True,
        next_node=first_stage,
    )

    redact_node = ToolNode(
        tool_function=_redact_for_log,
        input_keys=["__state__"],
        output_key=None,
        next_node=injection_guard,
    )

    intake_node = ToolNode(
        tool_function=_validate_intake,
        input_keys=["__state__"],
        output_key=None,
        next_node=redact_node,
    )

    return intake_node, ledger
