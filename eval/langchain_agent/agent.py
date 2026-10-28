"""
LangGraph implementation of the P&C risk triage pipeline (see SPEC.md).

Fan-out/fan-in uses plain multi-edge LangGraph parallelism (three add_edge
calls from one node, three add_edge calls into the join node) rather than
the Send() API: Send() is for mapping a node over a dynamically-sized list
at runtime, which this isn't -- there are always exactly three named,
fixed branches, so plain parallel edges are the idiomatic fit and the
state reducer (operator.add on branch_results) is what actually performs
the fan-in merge.
"""
import json
import operator
from typing import TypedDict, Annotated, List, Dict, Optional

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from compliance import (
    redact_pii,
    check_prompt_injection,
    PromptInjectionError,
    compute_oversight_level,
    AuthorityLedger,
    worst_severity,
)

MODEL_NAME = "qwen2.5:14b"
OLLAMA_BASE_URL = "http://localhost:11434"

llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME, temperature=0)


class IntakeRejected(Exception):
    pass


class RiskTriageState(TypedDict, total=False):
    raw_submission: str
    redacted_for_log: str
    branch_results: Annotated[List[Dict], operator.add]
    consolidated_risk_level: str
    consolidated_confidence: float
    consolidated_summary: str
    oversight_level: str
    oversight_reasoning: list
    final_decision: str
    ledger_record: dict


PROPERTY_PROMPT = """You are a property and catastrophe risk underwriter. Assess ONLY the \
physical property risk described below: flood/catastrophe exposure, construction type and \
age, and prior-claims history. Ignore financial standing and regulatory/sanctions matters \
entirely -- those are scored by separate specialists. Pay close attention to negation \
("NOT in a flood zone", "no history of...") -- a negated risk factor is not a risk.

Respond with ONLY a JSON object, no other text:
{{"risk_level": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL", "confidence": <0.0-1.0>, "finding": "<one sentence>"}}

Submission:
{submission}"""

FINANCIAL_PROMPT = """You are a financial/credit risk underwriter. Assess ONLY the financial \
exposure described below: sum insured relative to the applicant's financial standing, trading \
history, and credit profile. Ignore physical property condition and regulatory/sanctions \
matters entirely -- those are scored by separate specialists.

Respond with ONLY a JSON object, no other text:
{{"risk_level": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL", "confidence": <0.0-1.0>, "finding": "<one sentence>"}}

Submission:
{submission}"""

REGULATORY_PROMPT = """You are a regulatory/compliance risk officer. Assess ONLY sanctions, \
jurisdiction, and AML/compliance flags in the submission below. Ignore physical property \
condition and financial exposure entirely -- those are scored by separate specialists.

Respond with ONLY a JSON object, no other text:
{{"risk_level": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL", "confidence": <0.0-1.0>, "finding": "<one sentence>"}}

Submission:
{submission}"""


def _call_branch(dimension: str, prompt_template: str, submission: str) -> Dict:
    prompt = prompt_template.format(submission=submission)
    response = llm.invoke(prompt)
    content = response.content.strip()
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        parsed = json.loads(content[start:end])
        risk_level = str(parsed.get("risk_level", "")).upper()
        if risk_level not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            raise ValueError(f"invalid risk_level: {risk_level!r}")
        confidence = float(parsed.get("confidence", 0.0))
        finding = str(parsed.get("finding", ""))
    except Exception as e:
        # Fail safe: any parse/validation failure fails toward the MOST
        # cautious outcome (CRITICAL, confidence 0.0), never a silent
        # default to a clean result.
        risk_level, confidence, finding = (
            "CRITICAL",
            0.0,
            f"PARSE_ERROR: {e} -- raw_model_output={content[:200]!r}",
        )
    return {
        "dimension": dimension,
        "risk_level": risk_level,
        "confidence": confidence,
        "finding": finding,
    }


def intake_node(state: RiskTriageState) -> Dict:
    text = state.get("raw_submission")
    if not text or not text.strip():
        raise IntakeRejected("empty or missing submission")
    return {}


def pii_redact_node(state: RiskTriageState) -> Dict:
    return {"redacted_for_log": redact_pii(state["raw_submission"])}


def injection_guard_node(state: RiskTriageState) -> Dict:
    check_prompt_injection(state["raw_submission"])
    return {}


def property_risk_node(state: RiskTriageState) -> Dict:
    return {"branch_results": [_call_branch("property_risk", PROPERTY_PROMPT, state["raw_submission"])]}


def financial_risk_node(state: RiskTriageState) -> Dict:
    return {"branch_results": [_call_branch("financial_risk", FINANCIAL_PROMPT, state["raw_submission"])]}


def regulatory_risk_node(state: RiskTriageState) -> Dict:
    return {"branch_results": [_call_branch("regulatory_risk", REGULATORY_PROMPT, state["raw_submission"])]}


def consolidate_node(state: RiskTriageState) -> Dict:
    branches = state["branch_results"]
    levels = [b["risk_level"] for b in branches]
    worst = worst_severity(levels)
    avg_conf = sum(b["confidence"] for b in branches) / len(branches)
    lines = [
        f"  {b['dimension']:<16}: {b['risk_level']:<8} (confidence={b['confidence']:.2f}) -- {b['finding']}"
        for b in branches
    ]
    summary = "Branch findings (consolidated risk_level is the WORST of the three, not an average):\n" + "\n".join(lines)
    return {
        "consolidated_risk_level": worst,
        "consolidated_confidence": avg_conf,
        "consolidated_summary": summary,
    }


def oversight_node(state: RiskTriageState) -> Dict:
    result = compute_oversight_level("PROPERTY_RISK_SUBMISSION", state["consolidated_confidence"])
    return {"oversight_level": result["oversight_level"], "oversight_reasoning": result["reasoning"]}


def make_human_decision_node(ledger: AuthorityLedger, human_decision_provider):
    def human_decision_node(state: RiskTriageState) -> Dict:
        decision, rationale = human_decision_provider(state)
        rec = ledger.record(
            action_description="Bind or refer property risk submission",
            stakeholder_id="eval-harness-auto",
            stakeholder_role="AUTOMATED_EVAL_STAKEHOLDER",
            decision=decision,
            rationale=rationale,
            risk_score={
                "consolidated_risk_level": state["consolidated_risk_level"],
                "oversight_level": state["oversight_level"],
            },
        )
        return {"final_decision": decision.upper(), "ledger_record": rec}

    return human_decision_node


def auto_decision_node(state: RiskTriageState) -> Dict:
    # Reachable only when oversight_level != PRE_EXECUTION. With this spec's
    # single registered policy (reversibility=False), that never actually
    # happens -- same intentional finding as the Lar side's original demo:
    # an irreversible action always requires sign-off, confidence doesn't
    # buy its way out of oversight. Kept as a real, reachable code path
    # (not stubbed) in case a second, reversible policy is added later.
    risk_level = state["consolidated_risk_level"]
    decision = "bind" if risk_level in ("LOW", "MEDIUM") else "refer"
    return {"final_decision": decision.upper()}


def eval_harness_decision_provider(state: RiskTriageState):
    """Auto-resolver used for the non-interactive eval harness run. In
    production this callback is replaced by a real human reviewer's input --
    the pause point and the ledger write are real either way."""
    risk_level = state["consolidated_risk_level"]
    decision = "bind" if risk_level in ("LOW", "MEDIUM") else "refer"
    rationale = (
        f"Auto-resolved for eval harness (no live human reviewer attached). "
        f"AI-recommended decision based on consolidated_risk_level={risk_level}: {decision}. "
        f"In production this pause point is a real human sign-off, not this callback."
    )
    return decision, rationale


def build_graph(ledger: AuthorityLedger, human_decision_provider):
    graph = StateGraph(RiskTriageState)

    graph.add_node("intake", intake_node)
    graph.add_node("pii_redact", pii_redact_node)
    graph.add_node("injection_guard", injection_guard_node)
    graph.add_node("property_risk", property_risk_node)
    graph.add_node("financial_risk", financial_risk_node)
    graph.add_node("regulatory_risk", regulatory_risk_node)
    graph.add_node("consolidate", consolidate_node)
    graph.add_node("oversight", oversight_node)
    graph.add_node("human_decision", make_human_decision_node(ledger, human_decision_provider))
    graph.add_node("auto_decision", auto_decision_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "pii_redact")
    graph.add_edge("pii_redact", "injection_guard")

    # Fan-out: three parallel branches from the same node.
    graph.add_edge("injection_guard", "property_risk")
    graph.add_edge("injection_guard", "financial_risk")
    graph.add_edge("injection_guard", "regulatory_risk")

    # Fan-in: LangGraph waits for all three predecessors before running
    # consolidate (join semantics), merging branch_results via the
    # Annotated[..., operator.add] reducer on the state schema.
    graph.add_edge("property_risk", "consolidate")
    graph.add_edge("financial_risk", "consolidate")
    graph.add_edge("regulatory_risk", "consolidate")

    graph.add_edge("consolidate", "oversight")

    def route_oversight(state: RiskTriageState) -> str:
        return "human_decision" if state["oversight_level"] == "PRE_EXECUTION" else "auto_decision"

    graph.add_conditional_edges(
        "oversight",
        route_oversight,
        {"human_decision": "human_decision", "auto_decision": "auto_decision"},
    )
    graph.add_edge("human_decision", END)
    graph.add_edge("auto_decision", END)

    return graph.compile()
