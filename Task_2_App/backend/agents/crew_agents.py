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

from config import GROQ_API_KEY, GROQ_TEXT_MODEL, GROQ_MID_MODEL, GROQ_FAST_MODEL

logger = logging.getLogger(__name__)

# crewai injects 'cache_breakpoint' into message dicts before calling litellm.
# Groq rejects any message with unknown properties — strip it before the HTTP call.
_orig_completion = litellm.completion


def _patched_completion(*args, **kwargs):
    for msg in kwargs.get("messages", []):
        msg.pop("cache_breakpoint", None)
    return _orig_completion(*args, **kwargs)


litellm.completion = _patched_completion


def _make_llm(speed: str = "heavy") -> LLM:
    _models = {
        "heavy": GROQ_TEXT_MODEL,   # openai/gpt-oss-120b — ShuntAnalyst,GuidancePlannner
        "mid":   GROQ_MID_MODEL,    # llama-3.3-70b-versatile — Interpreter, Circuit, Planner
        "fast":  GROQ_FAST_MODEL,   # llama-3.1-8b-instant — JSON formatter
    }
    return LLM(
        model=f"groq/{_models[speed]}",
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )


# ── Agent 1 (ported from Task-1 ClinicalInterpreter) ─────────────────────────

def make_clinical_interpreter() -> Agent:
    """
    Reads confirmed clips, VLM frame annotation, and scan history.
    Determines whether the accumulated evidence is clinically sufficient to
    describe the circuit so far, and identifies any ambiguous or missing clip evidence.
    Uses mid-tier model — synthesis task, not strict rule reasoning.
    """
    return Agent(
        role="CHIVA Clinical Interpreter",
        goal=(
            "Given confirmed EP/RP clips, a VLM frame annotation, and a scan history summary, "
            "assess the clinical soundness of each confirmed clip. "
            "Identify which clips are unambiguous and which (if any) may be artefactual or misclassified "
            "(e.g. AASV labelled as GSV, or a clip recorded at a posY inconsistent with the anatomy). "
            "The VLM annotation includes: fascial layer visibility, N-class vessel presence "
            "(N1=deep vein below fascia, N2=saphenous trunk inside fascia, N3=tributary above fascia), "
            "and vein_types listing specific veins identified (e.g. GSV, FV, Tributary). "
            "Cross-reference the VLM annotation: does the visible anatomy support or conflict with the clip list? "
            "Do NOT infer or project what clips should be present — report only what is confirmed. "
            "If the VLM annotation indicates poor image quality or an off-axis view, flag this "
            "explicitly — probe repositioning should take priority over any clip assessment. "
            "Max 100 words."
        ),
        backstory=(
            "Expert CHIVA vascular surgeon with 20 years of duplex scanning experience. "
            "Fluent in EP/RP, N1/N2/N3, SFJ, SPJ, GSV, SSV notation. "
            "Understands annotated duplex frame overlays: GREEN oval outline = N2 vessel (saphenous trunk inside fascia), "
            "YELLOW oval outline = N3 vessel (tributary above fascia), BLUE oval outline = N1 vessel (deep vein below fascia), "
            "two yellow horizontal lines = fascia boundary separating superficial from deep compartment. "
            "Never fabricates flow events not explicitly stated in the clip list. "
            "Knows that EP N1→N2 at SFJ requires BOTH Valsalva AND Paranà to be positive; "
            "that AASV is N3 not N2; and that an elimination test is required before "
            "distinguishing Type 3 from Type 1+2."
        ),
        llm=_make_llm("mid"),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
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
            "Knows the full minimum clip set for every CHIVA type:\n"
            "TYPE 1:   EP N1→N2 (SFJ or Hunterian) + RP N2→N1. NO N3 clips. "
            "SFJ entry confirmed; GSV trunk refluxes; no escape to tributaries.\n"
            "TYPE 2A:  EP N2→N3 + RP N3→N1. NO EP N1→N2. NO EP N2→N2. NO RP N2→N1. "
            "SFJ competent; GSV escapes into tributary; tributary re-enters deep system.\n"
            "TYPE 2B:  EP N2→N2 + RP N3→N1. NO EP N1→N2. NO RP N2→N1. "
            "Perforator feeds system; tributary re-enters deep; no GSV trunk reflux.\n"
            "TYPE 2C:  (EP N2→N2 or EP N2→N3) + RP N3 + RP N2→N1. NO EP N1→N2. "
            "Perforator entry with secondary GSV trunk reflux; SFJ still competent.\n"
            "TYPE 3:   EP N1→N2 + EP N2→N3 + RP N3→N1 + RP N2→N1 + elimTest=No Reflux. "
            "Elimination test is MANDATORY. EP N1→N2 + EP N2→N3 + RP N3→N1 alone (no RP N2→N1) is NOT confirmed Type 3 — "
            "the surgeon must confirm trunk reflux (RP N2→N1) and then perform the elimination test.\n"
            "TYPE 1+2: EP N1→N2 + EP N2→N3 + RP N3→N1 + RP N2→N1 + elimTest=Reflux. "
            "Same clips as Type 3 — only the elimination test result distinguishes them.\n"
            "TYPE 4:   EP N1→N3 + RP N2→N1. "
            "N2 trunk actively refluxes back to N1. N2 is in the RETURN path.\n"
            "TYPE 5:   EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1. NO RP N2→N1. "
            "Blood loops N1→N3→N2→N3→N1; GSV loops but never directly drains to N1.\n"
            "TYPE 6:   EP N1→N3 + RP N3→N1. NO N2 involvement at all. "
            "Tributary returns straight to N1 via a second perforator; GSV completely bypassed.\n"
            "CRITICAL: For EP N1→N3 types — Type4=RP N2→N1 present; Type5=EP N2→N3 present, no RP N2→N1; Type6=no N2. "
            "Never declares a type confirmed without the full minimum clip set."
        ),
        llm=_make_llm("heavy"),  # strict rule reasoning — needs 120B
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


# ── Agent 3 (new for Task-2) ──────────────────────────────────────────────────

def make_circuit_analyst(tools: list | None = None) -> Agent:
    """
    Reads the Q1-Q4 status and scan history to determine the current examination
    state and the specific next target zone — with full CHIVA progression awareness.
    """
    return Agent(
        role="CHIVA Circuit Analyst",
        goal=(
            "Identify the current CHIVA examination state and the specific next target zone. "
            "Match what has been found (clips) and what has been visited (scan history) "
            "to the CHIVA flow: Q1 open → find entry point → Q2 → trunk reflux → "
            "Q3 → escape → Q4 → re-entry. "
            "If no clips found after scanning all major zones, direct the probe back "
            "to SFJ/groin to re-examine. Always name a specific zone. ≤70 words."
        ),
        backstory=(
            "Senior CHIVA examination specialist. You know exactly what anatomical "
            "structure to look for at every zone on the leg.\n\n"
            "ZONE ANATOMY — what each zone contains and where to find it:\n"
            "  SFJ/groin (posY 0.04–0.07): GSV meets CFV at the groin crease, "
            "anterior-medial surface. Terminal valve. Find: SFJ junction.\n"
            "  Upper thigh (posY 0.08–0.20): GSV in saphenous fascial compartment, "
            "medial surface. Find: GSV inside the 'saphenous eye' in transverse.\n"
            "  Hunterian (proximal thigh) (posY 0.21–0.33): Hunterian perforators connect FV to GSV in Hunter's canal, "
            "medial surface. KEY site for EP N1→N2 when SFJ is competent. Find: GSV-to-FV perforator junction on medial proximal thigh.\n"
            "  Dodd (distal thigh) (posY 0.34–0.47): Dodd perforators connect FV to GSV just above the knee, "
            "medial surface. Find: Dodd perforator at GSV junction on medial distal thigh.\n"
            "  Popliteal/SPJ (posY 0.48–0.57): SSV meets popliteal vein, "
            "posterior surface. Find: SPJ junction in popliteal fossa.\n"
            "  Calf (posY 0.58–0.88): GSV on medial surface, SSV posterior. "
            "Find: re-entry perforators and tributary junctions.\n"
            "  Ankle (posY 0.89–1.00): GSV at medial malleolus. "
            "Find: distal perforators and GSV.\n\n"
            "CORE PRINCIPLE: The task description tells you whether the probe is "
            "inside the Q1 corridor or not. When inside (posY 0.04–0.57), "
            "describe what to find at the CURRENT zone. "
            "When outside (posY > 0.57), identify which Q1 zone to route to next. "
            "A zone visited without a clip is not cleared — but the probe is there now, "
            "so guide it within the zone before suggesting it move."
        ),
        llm=_make_llm("heavy"),  # core navigation brain — zone routing + examination state reasoning
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


# ── Agent 4 (new for Task-2) ──────────────────────────────────────────────────

def make_navigation_planner(tools: list | None = None) -> Agent:
    """
    Decides where to move the probe next or what zone to scan, based on the
    circuit analyst's open-Q output and the current probe position.
    Has access to get_zone_protocol and get_vein_examination_objectives tools
    so it can look up the correct protocol for the TARGET zone before issuing
    its movement command.
    """
    return Agent(
        role="CHIVA Navigation Planner",
        goal=(
            "Give the trainee ONE probe movement command in ≤15 words. "
            "Every command starts with 'Move' and MUST contain:\n"
            "  1. A DIRECTION: proximally / distally / medially / laterally / "
            "posteriorly / anteriorly\n"
            "  2. A TARGET that always names BOTH the anatomical structure AND its location "
            "on the leg — never just 'GSV' alone, always 'GSV in saphenous compartment at upper thigh' "
            "or 'SFJ junction at groin crease' or 'Hunterian perforator on medial proximal thigh'.\n"
            "Example valid commands:\n"
            "  'Move medially toward GSV in saphenous compartment at upper thigh'\n"
            "  'Move proximally toward SFJ junction at groin crease'\n"
            "  'Move posteriorly toward SPJ junction in popliteal fossa'\n"
            "  'Move distally to Dodd perforator on medial distal thigh'\n"
            "  'Move medially toward GSV at medial malleolus (ankle)'"
        ),
        backstory=(
            "You are the probe navigator for a real-time CHIVA duplex examination. "
            "Your output is always a movement command: direction + destination. "
            "You have access to get_zone_protocol — use it to look up the examination "
            "protocol for the zone you are directing the probe to. This ensures your "
            "command is grounded in the correct clinical protocol for the target zone, "
            "not the current zone. Call get_zone_protocol with the target zone name "
            "before composing your movement command whenever routing to a new zone.\n"
            "When the probe needs to stay in the current zone, give micro-navigation "
            "toward a specific structure (e.g. 'Move medially toward GSV trunk'). "
            "When the probe needs to change zones, give a routing command "
            "(e.g. 'Move proximally to SFJ/groin'). "
            "You speak in anatomical terms: GSV, SFJ junction, saphenous compartment, "
            "Hunterian perforator, Dodd perforator, SPJ junction, popliteal vein, "
            "re-entry perforator. You never refer to clinical maneuvers."
        ),
        llm=_make_llm("mid"),
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )



# ── Agent 5 (new for Task-2, role of GeneralMedicalAssistant in guidance) ────

def make_guidance_specialist() -> Agent:
    """
    Converts the navigation plan into a single ≤12-word JSON probe-movement
    instruction conforming to the streaming UI format.
    """
    return Agent(
        role="CHIVA Real-Time Guidance Specialist",
        goal=(
            "Convert the Navigation Planner's instruction into a JSON object: "
            '{"guidance": "<≤15 words>", "action": "move"|"maneuver"|"complete"}. '
            "guidance is navigation only — where to move the probe or what zone to scan. "
            "action is determined by the Shunt Analyst: "
            "maneuver if elim_required=true, complete if confirmed=true, move otherwise. "
            "some maneuver words in guidance: Paranà, Valsalva, Doppler, compress, squeeze, "
            "maneuver, apply, perform, EP, RP, N1, N2, N3, Q1, Q2, Q3, Q4, shunt."
        ),
        backstory=(
            "You are the output formatter for a real-time ultrasound navigation system. "
            "The Navigation Planner has already decided where the probe should go. "
            "Your job is to package that decision as clean JSON. "
            "The surgeon is a trained vascular specialist — they know what to do at each zone. "
            "Your guidance just tells them where to point the probe, nothing more."
        ),
        llm=_make_llm("heavy"),  # final output — must correctly parse task2 flags and copy task4 exactly
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )