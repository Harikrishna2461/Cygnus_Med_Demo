"""
CrewAI pipeline for Task-2 streaming guidance.

ONE Crew, 5 agents, 5 tasks linked via context=[prev_task].
CrewAI's sequential process passes each task's output as context to downstream
tasks automatically — this is what makes agents actually interact with each other.

Task graph (what each agent sees):
  task1 (ClinicalInterpreter) — full state: clips + VLM + history
    └─context→ task2 (ShuntAnalyst)       — sees task1 output
      └─context→ task3 (CircuitAnalyst)   — sees task1 + task2 output
        └─context→ task4 (NavigationPlanner) — sees task3 output + protocol
          └─context→ task5 (GuidanceSpecialist) — sees task2+task3+task4 outputs

The utility modules (history_agent, q_state_agent, protocol_agent, vlm_agent)
are plain Python — they build the context blocks fed INTO task1's description.
"""
from __future__ import annotations

import json
import logging

from crewai import Crew, Process, Task

from agents.crew_agents import (
    make_circuit_analyst,
    make_clinical_interpreter,
    make_guidance_specialist,
    make_navigation_planner,
    make_shunt_analyst,
)

logger = logging.getLogger(__name__)


def _extract_json_str(raw: str) -> str:
    """Strip markdown fences and return the first JSON object."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(ln for ln in cleaned.splitlines() if not ln.startswith("```"))
    s, e = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[s : e + 1] if s != -1 and e != -1 else cleaned


def run_guidance_crew(
    state_message: str,
    region: str,
    pos_y: float,
    surface: str,
    leg: str,
    clips: list[dict],
    vlm_summary: str,
    history_summary: str,
    q_state: str,
    protocol_text: str,
    pos_x: float | None = None,
    is_front: bool | None = None,
) -> tuple[str, str, str]:
    """
    Run the 5-agent guidance crew as a single CrewAI sequential Crew.
    Returns (guidance_text, raw_log, action).

    Agents interact through CrewAI's context mechanism: each task receives
    the outputs of its upstream tasks automatically — agents build on each
    other's reasoning without manual string injection.
    """
    pos_x_str = f" | posX={pos_x:.2f}" if pos_x is not None else ""
    front_str = f" | is_front={'yes' if is_front else 'no'}" if is_front is not None else ""

    clips_text = "\n".join(
        f"  {c.get('flow')} {c.get('from_type')}→{c.get('to_type')} "
        f"posY={float(c.get('pos_y_ratio', 0)):.2f} {c.get('leg', '?')} leg"
        + (f" [elimTest={c['elimination_test']}]" if c.get("elimination_test") else "")
        for c in clips
    ) or "  None confirmed yet."

    # ── Create all 5 agents ───────────────────────────────────────────────────
    interpreter = make_clinical_interpreter()
    analyst     = make_shunt_analyst()
    circuit     = make_circuit_analyst()
    planner     = make_navigation_planner()
    specialist  = make_guidance_specialist()

    # ── Task 1: Clinical Interpreter ──────────────────────────────────────────
    task1 = Task(
        description=(
            "You are the first agent in a 5-agent CHIVA examination pipeline. "
            "Assess the clinical picture from the full examination state below.\n\n"
            f"{state_message}\n\n"
            "In max 100 words:\n"
            "1. List which confirmed clips are unambiguous and what each establishes.\n"
            "2. Flag any clip that may be artefactual (e.g. AASV labelled as GSV).\n"
            "3. State whether the VLM frame supports or conflicts with the clip list.\n"
            "4. Identify clips expected but absent given the developing circuit.\n"
            "Do NOT fabricate flow events absent from CONFIRMED FINDINGS."
        ),
        expected_output=(
            "Clinical interpretation (≤100 words): clip status, VLM alignment, "
            "expected-but-absent clips."
        ),
        agent=interpreter,
    )

    # ── Task 2: Shunt Analyst (sees task1) ───────────────────────────────────
    task2 = Task(
        description=(
            "You are the second agent. The Clinical Interpreter (previous agent) has "
            "assessed the clips. Using that assessment and the raw clip list below, "
            "classify the developing CHIVA shunt type.\n\n"
            f"RAW CONFIRMED CLIPS ({len(clips)} total):\n{clips_text}\n\n"
            "In max 80 words:\n"
            "1. Most likely developing shunt type (I, 2A, 2B, 2C, 3, 4, 5, 6, or undetermined) "
            "and the supporting clip pattern.\n"
            "2. Specific additional clips needed to confirm or exclude each candidate type.\n"
            "3. Whether an elimination test is required before the type can be determined."
        ),
        expected_output=(
            "Shunt analysis (≤80 words): developing type, supporting pattern, "
            "missing clips, elimination test flag."
        ),
        agent=analyst,
        context=[task1],
    )

    # ── Task 3: Circuit Analyst (sees task1 + task2) ──────────────────────────
    task3 = Task(
        description=(
            "You are the third agent. The Clinical Interpreter and Shunt Analyst have "
            "assessed the clips and classified the developing type. "
            "Use their outputs alongside the Q1-Q4 status and scan history below "
            "to identify the open diagnostic question and target zone.\n\n"
            f"Q1-Q4 STATUS:\n{q_state}\n\n"
            f"SCAN HISTORY:\n{history_summary}\n\n"
            "In max 60 words:\n"
            "1. Which Q1-Q4 question is currently open?\n"
            "2. Which anatomical zone (name + posY band) must be examined to answer it?\n"
            "3. Is an elimination test or circuit-complete declaration required right now?"
        ),
        expected_output=(
            "Circuit analysis (≤60 words): open Q, target zone (name + posY band), "
            "immediate action if any."
        ),
        agent=circuit,
        context=[task1, task2],
    )

    # ── Task 4: Navigation Planner (sees task3) ───────────────────────────────
    task4 = Task(
        description=(
            "You are the fourth agent. The Circuit Analyst has identified the open "
            "diagnostic question and target zone. Using that output and the zone-specific "
            "examination protocol below, produce a precise navigation plan.\n\n"
            f"CURRENT PROBE POSITION: region={region}, posY={pos_y:.2f}, "
            f"surface={surface}, leg={leg}{pos_x_str}{front_str}\n\n"
            f"EXAMINATION PROTOCOL:\n{protocol_text}\n\n"
            "In max 50 words, specify:\n"
            "- Target posY band (e.g. 0.21–0.35 for Hunterian)\n"
            "- Probe surface (anterior-medial / medial / posterior / lateral)\n"
            "- Named anatomical target (e.g. 'mid-thigh GSV in saphenous eye')\n"
            "- Direction word (distally / proximally / medially / posteriorly / "
            "laterally / transversely)\n"
            "- Maneuver (Paranà / Valsalva / squeezing / transverse scan)"
        ),
        expected_output=(
            "Navigation plan (≤50 words): target posY band, surface, anatomical target, "
            "direction word, maneuver."
        ),
        agent=planner,
        context=[task3],
    )

    # ── Task 5: Guidance Specialist (sees task2 + task3 + task4) ─────────────
    task5 = Task(
        description=(
            "You are the fifth and final agent. The Shunt Analyst, Circuit Analyst, "
            "and Navigation Planner have all contributed. Synthesise their outputs into "
            "a single JSON probe-movement instruction.\n\n"
            "Output ONLY valid JSON (no markdown, no explanation):\n"
            '  {"guidance": "<single imperative ≤12 words>", "action": "move"}\n\n'
            "action values:\n"
            "  'move'     — default for all probe navigation\n"
            "  'maneuver' — ONLY if circuit analysis explicitly requires elimination test\n"
            "  'complete' — ONLY if circuit analysis declares full circuit confirmed\n\n"
            "FORBIDDEN in guidance text: EP, RP, N1, N2, N3, reflux, Q1, Q2, Q3, Q4, "
            "confirmed, findings, diagnostic, shunt, Given, Since, As the, Currently.\n"
            "guidance MUST contain one direction word: distally / proximally / medially / "
            "posteriorly / laterally / transversely / deeper."
        ),
        expected_output='Valid JSON only: {"guidance": "...", "action": "move"}',
        agent=specialist,
        context=[task2, task3, task4],
    )

    # ── Single Crew — one kickoff, agents interact via context ────────────────
    crew = Crew(
        agents=[interpreter, analyst, circuit, planner, specialist],
        tasks=[task1, task2, task3, task4, task5],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff()
    except Exception as exc:
        logger.error("[CrewAI] Crew kickoff failed: %s", exc)
        return (
            "Continue scanning distally to locate anatomical junction",
            f"[crew-error] {exc}",
            "move",
        )

    # ── Extract final output (task5) and intermediate outputs for logging ─────
    raw = result.raw if hasattr(result, "raw") else str(result)

    # tasks_output is a list of TaskOutput objects, one per task
    tasks_out = getattr(result, "tasks_output", [])
    def _t(i: int) -> str:
        if i < len(tasks_out):
            o = tasks_out[i]
            return (o.raw if hasattr(o, "raw") else str(o))[:80]
        return "—"

    combined_raw = (
        f"[interpret] {_t(0)} | "
        f"[shunt] {_t(1)} | "
        f"[circuit] {_t(2)} | "
        f"[nav] {_t(3)} | "
        f"[guidance] {raw[:80]}"
    )

    # ── Parse JSON from task5 output ──────────────────────────────────────────
    guidance = "Continue scanning distally to locate anatomical junction"
    action = "move"

    try:
        parsed = json.loads(_extract_json_str(raw))
        guidance = parsed.get("guidance", guidance)
        action = parsed.get("action", "move")
        if action not in ("move", "maneuver", "complete"):
            action = "move"
    except json.JSONDecodeError:
        logger.warning("[CrewAI] Could not parse GuidanceSpecialist output: %.200s", raw)

    return guidance, combined_raw, action
