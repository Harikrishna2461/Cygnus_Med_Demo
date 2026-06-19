"""
CrewAI agent definitions for the Task-2 streaming guidance pipeline.

Five agents — three ported from Task-1, two new for real-time navigation:

  Task-1 ports (provide clinical context upstream):
    make_clinical_interpreter()  - reads clips + VLM + history; assesses sufficiency of circuit picture
    make_shunt_analyst()         - takes interpretation; classifies developing shunt type + missing evidence
    make_circuit_analyst()       - takes shunt analysis; derives open Q1-Q4 question + target anatomical zone

  Task-2 navigation-specific (downstream, act on clinical context):
    make_navigation_planner()    - takes circuit analysis; selects posY band, surface, target, maneuver
    make_guidance_specialist()   - takes navigation plan; outputs final JSON probe-movement instruction

Each agent's output feeds the next agent's task description (CrewAI sequential context passing).
Each factory creates a fresh Agent; do not share instances across requests.

Requires Python >=3.10,<=3.13. Run with Task-1's .venv (Python 3.12):
  C:\\Users\\Krish\\Downloads\\Cygnus_Med_Demo\\Task_1_App\\.venv\\Scripts\\python.exe
"""
from __future__ import annotations

import logging

import litellm
from crewai import Agent, LLM

from config import GROQ_API_KEY, GROQ_TEXT_MODEL

logger = logging.getLogger(__name__)

# crewai injects 'cache_breakpoint' into message dicts before calling litellm.
# Groq rejects any message with unknown properties — strip it before the HTTP call.
_orig_completion = litellm.completion


def _patched_completion(*args, **kwargs):
    for msg in kwargs.get("messages", []):
        msg.pop("cache_breakpoint", None)
    return _orig_completion(*args, **kwargs)


litellm.completion = _patched_completion


def _make_llm() -> LLM:
    return LLM(
        model=f"groq/{GROQ_TEXT_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )


# ── Agent 1 (ported from Task-1 ClinicalInterpreter) ─────────────────────────

def make_clinical_interpreter() -> Agent:
    """
    Reads confirmed clips, VLM frame annotation, and scan history.
    Determines whether the accumulated evidence is clinically sufficient to
    describe the circuit so far, and identifies any ambiguous or missing clip evidence.
    """
    return Agent(
        role="CHIVA Clinical Interpreter",
        goal=(
            "Given confirmed EP/RP clips, a VLM frame annotation, and a scan history summary, "
            "assess whether the accumulated duplex findings sufficiently characterise the "
            "haemodynamic circuit so far. Identify which clips are unambiguous, which are "
            "absent but expected, and whether the VLM frame shows anything that modifies the "
            "clip interpretation. Produce a concise clinical interpretation (max 100 words)."
        ),
        backstory=(
            "Expert CHIVA vascular surgeon with 20 years of duplex scanning experience. "
            "Fluent in EP/RP, N1/N2/N3, SFJ, SPJ, GSV, SSV notation. "
            "Never fabricates flow events not explicitly stated in the clip list. "
            "Knows that EP N1→N2 at SFJ requires BOTH Valsalva AND Paranà to be positive; "
            "that AASV is N3 not N2; and that an elimination test is required before "
            "distinguishing Type 3 from Type 1+2."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )


# ── Agent 2 (ported from Task-1 ShuntAnalyst) ────────────────────────────────

def make_shunt_analyst() -> Agent:
    """
    Takes the clinical interpretation and classifies the developing CHIVA shunt type.
    Identifies what clip evidence is still needed to close or confirm the circuit.
    """
    return Agent(
        role="CHIVA Shunt Classification Specialist",
        goal=(
            "From a clinical interpretation of confirmed clips, classify the developing "
            "CHIVA shunt type (Type I, 2A, 2B, 2C, 3, 4, 5, 6, or undetermined). "
            "State the clip pattern that supports the classification and list the specific "
            "additional clips that would confirm or exclude each candidate type. "
            "Flag if an elimination test is required. Keep to 80 words."
        ),
        backstory=(
            "Senior vascular surgeon specialising in CHIVA haemodynamic surgery. "
            "Maps EP/RP clip patterns to Mendoza shunt types I–VI: "
            "Type 1 = EP N1→N2 + RP N2→N1, no escape; "
            "Type 2 = EP N2→N3 + RP N3→N1, no trunk entry; "
            "Type 3/1+2 = escape present, elimination test required; "
            "Type 4 = EP N1→N3 + RP N2→N1; Type 5 = EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1; "
            "Type 6 = EP N1→N3 + RP N3→N1, no trunk. "
            "Never declares a type complete without the minimum required clip set."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )


# ── Agent 3 (new for Task-2) ──────────────────────────────────────────────────

def make_circuit_analyst() -> Agent:
    """
    Takes the shunt classification output and Q1-Q4 status, then derives:
    - which diagnostic question (Q1-Q4) is currently open
    - the specific anatomical zone that must be examined to answer it
    """
    return Agent(
        role="CHIVA Circuit Analyst",
        goal=(
            "From a shunt classification and Q1-Q4 status block, determine which "
            "diagnostic question (Q1–Q4) is currently open and name the exact anatomical "
            "zone (with posY band) that must be examined to answer it. "
            "Also flag if an immediate action is required (elimination test or circuit complete). "
            "Keep to 60 words."
        ),
        backstory=(
            "CHIVA duplex examination specialist who reads shunt type progressions and "
            "maps them to the four diagnostic questions: "
            "Q1=entry point, Q2=trunk reflux, Q3=trunk escape, Q4=tributary re-entry. "
            "Knows posY landmarks: 0.04-0.09=SFJ, 0.21-0.35=Hunterian, "
            "0.40-0.55=SPJ/popliteal, 0.56-0.80=calf, 0.81-1.00=ankle."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )


# ── Agent 4 (new for Task-2) ──────────────────────────────────────────────────

def make_navigation_planner() -> Agent:
    """
    Takes the circuit analyst's open-Q output and the zone-specific protocol,
    then selects the exact posY band, probe surface, anatomical target, and maneuver.
    """
    return Agent(
        role="CHIVA Navigation Planner",
        goal=(
            "Given an open diagnostic question, the current probe position, and the "
            "zone-specific examination protocol, produce a specific navigation plan: "
            "target posY band, probe surface (anterior-medial/medial/posterior/lateral), "
            "named anatomical target, direction word, and the applicable maneuver "
            "(Paranà/Valsalva/squeezing/transverse scan). Keep to 50 words."
        ),
        backstory=(
            "Vascular surgeon who directs duplex examinations following the Adler 2022 "
            "standard sequence and Delfrate 2023 perforator protocol. "
            "Knows that BOTH Valsalva AND Paranà are required at SFJ and SPJ (Gianesini 2014), "
            "that Paranà is preferred over squeezing alone in the thigh and calf, "
            "and that pathological perforators require ≥500 ms outward flow AND ≥3.5 mm diameter."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )


# ── Agent 5 (new for Task-2, role of GeneralMedicalAssistant in guidance) ────

def make_guidance_specialist() -> Agent:
    """
    Synthesizes the navigation plan into a single ≤12-word JSON probe-movement
    instruction conforming to the streaming UI format.
    """
    return Agent(
        role="CHIVA Real-Time Guidance Specialist",
        goal=(
            "Synthesise a navigation plan into a single JSON object: "
            '{"guidance": "<single imperative ≤12 words>", "action": "move"|"maneuver"|"complete"}. '
            "guidance = one imperative sentence with a direction word + anatomical target, ≤12 words. "
            "action = 'move' (default), 'maneuver' (elimination test only), 'complete' (full circuit only). "
            "Forbidden in guidance text: EP, RP, N1, N2, N3, reflux, Q1-Q4, confirmed, findings, "
            "diagnostic, shunt, Given, Since, As the, Currently."
        ),
        backstory=(
            "Real-time ultrasound navigation AI embedded in a streaming guidance UI. "
            "Produces only crisp, single-sentence probe directions the surgeon can act on instantly. "
            "guidance must always include a direction word: "
            "distally / proximally / medially / posteriorly / laterally / transversely / deeper."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )
