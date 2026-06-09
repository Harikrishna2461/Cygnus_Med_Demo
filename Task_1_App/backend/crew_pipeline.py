"""
CrewAI-backed pipeline — same public interface as the originals.

Callers (routes) import from here exactly as they would from the originals:
  parse_nl_to_clips(...)
  build_conversational_response(...)
  classify_and_plan_ligation_with_llm(...)
  generate_general_response(...)   <- new, replaces the direct Groq call in routes/general.py

call_llm_fn is accepted for API compatibility but unused here — agents carry their own LLM.
If a CrewAI call fails the exception propagates so the caller can fall back to the original.

All prompts and parsing helpers are imported from the original modules — no duplication.
"""

import json
import logging
import re
from typing import Callable

from crewai import Crew, Process, Task

from crew_agents import (
    make_clinical_interpreter,
    make_general_medical_assistant,
    make_shunt_analyst,
)

# Reuse every prompt and utility from the originals — nothing is copied.
from nl_interpreter import (
    _CONVERSATIONAL_PROMPT,
    _NL_TO_CHIVA_PROMPT,
    _SUFFICIENCY_PROMPT,
    _build_accumulated_description,
    _clean_json,
    _has_no_reflux_statement,
)
from shunt_classification_and_ligation_llm import (
    _LEG_ORDER,
    _repair_and_parse,
    _retrieve_rag_context_for_ligation,
    build_ligation_prompt,
    build_shunt_classification_prompt,
)

logger = logging.getLogger(__name__)


# ── Shared task runner ────────────────────────────────────────────────────────

def _run_task(agent, description: str, expected_output: str, retries: int = 2) -> str:
    """Run a single-agent single-task Crew and return the raw text output.
    Retries up to `retries` times on any exception before re-raising."""
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(retries + 1):
        try:
            task = Task(description=description, expected_output=expected_output, agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff()
            return result.raw if hasattr(result, "raw") else str(result)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                logger.warning(f"[CrewAI] Task attempt {attempt + 1} failed ({e}), retrying...")
    raise last_exc


def _extract_json(raw: str) -> str:
    """
    More robust than _clean_json: strips markdown fences then extracts the
    first JSON object even if the agent prepended a sentence of explanation.
    """
    cleaned = _clean_json(raw)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        return cleaned[start:end]
    return cleaned


# ── Clinical Interpreter ──────────────────────────────────────────────────────

def parse_nl_to_clips(
    user_message: str,
    call_llm_fn: Callable = None,
    history: list[dict] | None = None,
) -> dict:
    """
    CrewAI equivalent of nl_interpreter.parse_nl_to_clips.
    Two-stage: sufficiency check → (if sufficient) CHIVA interpretation.
    """
    accumulated = _build_accumulated_description(history, user_message)
    agent = make_clinical_interpreter()

    # Stage 1: sufficiency check
    try:
        raw = _run_task(
            agent,
            description=_SUFFICIENCY_PROMPT.format(description=accumulated),
            expected_output='Valid JSON only: {"verdict": "sufficient"|"insufficient"|"question", "missing": "..."}',
        )
        check = json.loads(_extract_json(raw))
        verdict = check.get("verdict", "sufficient")
    except Exception as e:
        logger.error(f"[CrewAI] Sufficiency check failed: {e}. Falling through to CHIVA call.")
        verdict = "sufficient"
        check = {}

    if verdict == "question":
        return {
            "is_clinical": False,
            "sufficient_information": False,
            "missing_information": None,
            "interpretation": None,
            "clips": [],
        }

    if verdict == "insufficient":
        missing = check.get("missing") or (
            "Still need to know whether blood refluxes backward through the GSV trunk, "
            "whether it escapes into any tributary, and if so whether it also refluxes "
            "backward through that tributary."
        )
        return {
            "is_clinical": True,
            "sufficient_information": False,
            "missing_information": missing,
            "interpretation": None,
            "clips": [],
        }

    # Stage 2: CHIVA interpretation (only if verdict == "sufficient")
    try:
        raw = _run_task(
            agent,
            description=_NL_TO_CHIVA_PROMPT.format(description=accumulated),
            expected_output='Valid JSON only: {"interpretation": "...", "clips": [...]}',
        )
        result = json.loads(_extract_json(raw))
        if isinstance(result, dict) and "clips" in result:
            if _has_no_reflux_statement(accumulated) and result.get("clips"):
                before = len(result["clips"])
                result["clips"] = [c for c in result["clips"] if c.get("flow") != "RP"]
                after = len(result["clips"])
                if before != after:
                    logger.info(
                        f"[CrewAI] Stripped {before - after} hallucinated RP clip(s) "
                        f"— explicit no-reflux statement present."
                    )
            return {
                "is_clinical": True,
                "sufficient_information": True,
                "missing_information": None,
                "interpretation": result.get("interpretation"),
                "clips": result.get("clips", []),
            }
    except Exception as e:
        logger.error(f"[CrewAI] CHIVA interpretation failed: {e}")

    return {
        "is_clinical": False,
        "sufficient_information": False,
        "missing_information": None,
        "interpretation": None,
        "clips": [],
    }


def build_conversational_response(
    user_message: str,
    analysis_context: str,
    history: list[dict],
    call_llm_fn: Callable = None,
) -> str:
    """CrewAI equivalent of nl_interpreter.build_conversational_response."""
    history_lines = []
    for m in history[-8:]:
        role_label = "Clinician" if m.get("role") == "user" else "Assistant"
        history_lines.append(f"{role_label}: {m.get('content', '')[:400]}")

    prompt = _CONVERSATIONAL_PROMPT.format(
        analysis_context=analysis_context or "No prior clinical analysis available.",
        history="\n".join(history_lines) or "(start of conversation)",
        user_message=user_message.strip(),
    )
    agent = make_clinical_interpreter()
    try:
        return _run_task(
            agent,
            description=prompt,
            expected_output="Concise clinical response in plain text.",
        ).strip()
    except Exception as e:
        logger.error(f"[CrewAI] Conversational response failed: {e}")
        return (
            "I encountered an error generating a response. "
            "Please check your connection and try again."
        )


# ── Shunt Analysis ────────────────────────────────────────────────────────────

# Mirrored from shunt_classification_and_ligation_llm.py — these types skip ligation planning.
_NO_LIGATION_RESULT: dict[str, dict] = {
    "No shunt detected": {
        "ligation_steps": [],
        "clinical_rationale": "No pathological shunt identified. No surgical intervention required.",
        "additional_info_needed": [],
        "complications_contraindications": [],
        "followup_schedule": "",
        "chiva_approach": "",
        "confidence": 0.95,
    },
    "Undetermined": {
        "ligation_steps": ["Elimination test required before ligation planning can proceed"],
        "clinical_rationale": "Cannot distinguish Type 3 from Type 1+2 without the elimination test result.",
        "additional_info_needed": ["Perform elimination test and resubmit"],
        "complications_contraindications": [],
        "followup_schedule": "",
        "chiva_approach": "",
        "confidence": 0.0,
    },
}


def classify_and_plan_ligation_with_llm(
    clip_list: list[dict],
    call_llm_fn: Callable = None,
    retrieve_ligation_context_fn: Callable | None = None,
) -> dict:
    """
    CrewAI equivalent of shunt_classification_and_ligation_llm.classify_and_plan_ligation_with_llm.
    Per-leg: classification task → ligation planning task → merge into findings dict.
    """
    groups: dict[str, list[dict]] = {}
    if not clip_list:
        groups["Unspecified"] = []
    else:
        for c in clip_list:
            side = (c.get("legSide") or c.get("leg_side") or "Assessment").strip().capitalize()
            groups.setdefault(side, []).append(c)

    agent = make_shunt_analyst()
    findings: list[dict] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for leg_label, group in groups.items():

        # Step 1: Shunt Classification (no RAG)
        cls_prompt = build_shunt_classification_prompt(group, leg_label)
        try:
            raw = _run_task(
                agent,
                description=cls_prompt,
                expected_output=(
                    'Valid JSON only: {"shunt_type": "...", "confidence": 0.0, '
                    '"reasoning": [...], "needs_elim_test": false, ...}'
                ),
            )
            logger.info(f"[CrewAI] Classification raw response ({leg_label}): {raw!r}")
            classification = _repair_and_parse(_extract_json(raw))
            if not classification or "shunt_type" not in classification:
                logger.error(f"[CrewAI] Parsed classification: {classification!r}")
                raise RuntimeError(f"Unparseable classification response for {leg_label}")
        except Exception as e:
            logger.error(f"[CrewAI] Shunt classification failed for {leg_label}: {e}")
            raise RuntimeError(str(e)) from e

        shunt_type = classification.get("shunt_type", "Unknown")
        classification_usage = classification.pop("_llm_usage", {})

        # Step 2: Ligation Planning (with RAG; skipped for no-shunt / undetermined)
        if shunt_type in _NO_LIGATION_RESULT:
            ligation = dict(_NO_LIGATION_RESULT[shunt_type])
            ligation_usage: dict = {}
        else:
            rag_context = (
                _retrieve_rag_context_for_ligation(shunt_type, retrieve_ligation_context_fn)
                if retrieve_ligation_context_fn
                else "No RAG context available."
            )
            lig_prompt = build_ligation_prompt(shunt_type, group, rag_context, leg_label)
            try:
                raw = _run_task(
                    agent,
                    description=lig_prompt,
                    expected_output=(
                        'Valid JSON only: {"ligation_steps": [...], '
                        '"clinical_rationale": "...", "chiva_approach": "...", ...}'
                    ),
                )
                ligation = _repair_and_parse(_extract_json(raw))
                if not ligation or "ligation_steps" not in ligation:
                    raise RuntimeError(f"Unparseable ligation plan for {leg_label}")
                ligation_usage = ligation.pop("_llm_usage", {})
            except Exception as e:
                logger.error(f"[CrewAI] Ligation planning failed for {leg_label}: {e}")
                raise RuntimeError(str(e)) from e

        total_prompt_tokens += (
            classification_usage.get("prompt_tokens", 0)
            + ligation_usage.get("prompt_tokens", 0)
        )
        total_completion_tokens += (
            classification_usage.get("completion_tokens", 0)
            + ligation_usage.get("completion_tokens", 0)
        )

        ligation_steps = ligation.get("ligation_steps", [])
        findings.append({
            "leg": leg_label,
            "num_clips": len(group),
            "shunt_type": classification.get("shunt_type"),
            "assessment": classification.get("shunt_type"),
            "confidence": classification.get("confidence", 0.0),
            "chain_of_thought": classification.get("chain_of_thought", ""),
            "reasoning": classification.get("reasoning", []),
            "needs_elim_test": classification.get("needs_elim_test", False),
            "ask_branching": classification.get("ask_branching", False),
            "summary": classification.get("summary", ""),
            "ligation_steps": ligation_steps,
            "point_of_ligation": ligation_steps[0] if ligation_steps else "",
            "clinical_rationale": ligation.get("clinical_rationale", ""),
            "additional_info_needed": ligation.get("additional_info_needed", []),
            "complications_contraindications": ligation.get("complications_contraindications", []),
            "followup_schedule": ligation.get("followup_schedule", ""),
            "chiva_approach": ligation.get("chiva_approach", ""),
            "classification_llm_usage": classification_usage,
            "ligation_llm_usage": ligation_usage,
        })

    findings.sort(key=lambda f: _LEG_ORDER.get(f["leg"], 2))

    if not findings:
        raise RuntimeError("[CrewAI] Shunt classifier returned no findings")

    primary = findings[0]
    return {
        "findings": findings,
        "shunt_type": primary.get("shunt_type"),
        "confidence": primary.get("confidence", 0.0),
        "chain_of_thought": primary.get("chain_of_thought", ""),
        "reasoning": primary.get("reasoning", []),
        "ligation": primary.get("ligation_steps", []),
        "point_of_ligation": primary.get("point_of_ligation", ""),
        "summary": primary.get("summary", ""),
        "needs_elim_test": primary.get("needs_elim_test", False),
        "ask_branching": primary.get("ask_branching", False),
        "num_clips": len(clip_list),
        "num_findings": len(findings),
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }


# ── General Medical Chat ──────────────────────────────────────────────────────

def generate_general_response(system_prompt: str, user_prompt: str) -> str:
    """
    Drop-in for the direct groq_client.chat.completions.create call in routes/general.py.
    Combines system + user prompt and runs it through the GeneralMedicalAssistant agent.
    """
    agent = make_general_medical_assistant()
    combined = f"{system_prompt}\n\n{user_prompt}"
    try:
        return _run_task(
            agent,
            description=combined,
            expected_output="Plain text clinical response. No markdown. End with SOURCES.",
        ).strip()
    except Exception as e:
        logger.error(f"[CrewAI] General response failed: {e}")
        return f"Error generating response: {e}"
