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

import io
import json
import logging
import sys
import threading
from collections import Counter

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
    rejection_notes: list[str] | None = None,
    recent_guidance: list[str] | None = None,
    accepted_shunts: list[str] | None = None,
) -> tuple[str, str, str, bool, str, str]:
    """
    Run the 5-agent guidance crew as a single CrewAI sequential Crew.
    Returns (guidance_text, raw_log, action, shunt_found, shunt_type, shunt_evidence).

    Agents interact through CrewAI's context mechanism: each task receives
    the outputs of its upstream tasks automatically — agents build on each
    other's reasoning without manual string injection.
    """
    pos_x_str = f" | posX={pos_x:.2f}" if pos_x is not None else ""
    front_str = f" | is_front={'yes' if is_front else 'no'}" if is_front is not None else ""

    # Compute zone name deterministically — injected into Task 4 so the model
    # never has to reason from posY to zone name (a common error source).
    def _zone_name(y: float) -> str:
        if y <= 0.07: return "SFJ/groin"
        if y <= 0.20: return "upper thigh"
        if y <= 0.33: return "Hunterian (proximal thigh)"
        if y <= 0.47: return "Dodd (distal thigh)"
        if y <= 0.57: return "popliteal/SPJ"
        if y <= 0.88: return "calf"
        return "ankle"

    current_zone = _zone_name(pos_y)

    rejection_ctx = ""
    if rejection_notes:
        rejection_ctx = (
            "SURGEON FEEDBACK — PRIOR CLASSIFICATION REJECTED:\n"
            + "\n".join(f"  • {n}" for n in rejection_notes[-3:])
            + "\n\nRe-evaluate the evidence MORE critically than before. "
            "Do NOT re-confirm the rejected type without a meaningfully different clip set.\n\n"
        )

    accepted_ctx = ""
    if accepted_shunts:
        entries = ", ".join(accepted_shunts)
        accepted_ctx = (
            f"SURGEON-ACCEPTED SHUNTS: {entries}\n"
            "The surgeon has already accepted these shunt classifications and is actively "
            "searching for ADDITIONAL shunts. For any shunt type listed above, output "
            "confirmed=false — do NOT re-confirm an already-accepted type. "
            "Focus classification on NEW patterns from the clip set that have NOT yet been accepted.\n\n"
        )

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
            rejection_ctx +
            "You are the first agent in a 5-agent CHIVA examination pipeline. "
            "Assess the clinical picture from the full examination state below.\n\n"
            f"IMPORTANT: The 'Region' label in the state message is from the frontend "
            f"interface and may not perfectly match CHIVA anatomy (e.g. it may say "
            f"'Giacomini' or a non-standard label). Always use posY as the authoritative "
            f"position reference — posY {pos_y:.2f} = {current_zone}.\n\n"
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
            accepted_ctx +
            "You are the second agent and the sole authority on shunt type classification. "
            "The Clinical Interpreter has assessed the clips. "
            "Using that assessment and the raw clip list below, determine whether a CHIVA "
            "shunt type is confirmed and whether an elimination test is required right now.\n\n"
            f"RAW CONFIRMED CLIPS ({len(clips)} total):\n{clips_text}\n\n"
            "Output ONLY valid JSON (no markdown):\n"
            '  {"shunt_type": "1"|"2A"|"2B"|"2C"|"3"|"1+2"|"4"|"5"|"6"|"undetermined",\n'
            '   "confirmed": true|false,\n'
            '   "elim_required": true|false,\n'
            '   "evidence": "<one sentence: clip pattern that confirms, why unconfirmed, or why elim needed>"}\n\n'
            '"confirmed" is true ONLY when the FULL minimum clip set is present:\n'
            "  Type 1   → EP N1→N2 (SFJ or Hunterian) + RP N2→N1; NO N3 clips at all\n"
            "  Type 2A  → EP N2→N3 + RP N3→N1; NO EP N1→N2; NO EP N2→N2; NO RP N2→N1\n"
            "  Type 2B  → EP N2→N2 + RP N3→N1; NO EP N1→N2; NO RP N2→N1\n"
            "  Type 2C  → (EP N2→N2 or EP N2→N3) + RP N3 + RP N2→N1; NO EP N1→N2\n"
            "  Type 3   → EP N1→N2 + EP N2→N3 + RP N3→N1 + RP N2→N1 + elimTest=No Reflux\n"
            "  Type 1+2 → EP N1→N2 + EP N2→N3 + RP N3→N1 + RP N2→N1 + elimTest=Reflux\n"
            "  Type 4   → EP N1→N3 + RP N2→N1\n"
            "  Type 5   → EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1; NO RP N2→N1\n"
            "  Type 6   → EP N1→N3 + RP N3→N1; NO N2 clips whatsoever\n\n"
            f'"elim_required": if and only if ALL FOUR of these clips are in the {len(clips)}-clip '
            "confirmed list above: EP N1→N2, EP N2→N3, RP N3→N1, RP N2→N1 — AND no clip has "
            "an elimination_test value. If the confirmed clip count is fewer than 4, "
            "elim_required MUST be false. Do NOT infer clips that are not in the list above.\n\n"
            "KEY DIFFERENTIATOR for N1→N3-entry types:\n"
            "  Type 4: RP N2→N1 PRESENT\n"
            "  Type 5: EP N2→N3 PRESENT, NO RP N2→N1\n"
            "  Type 6: NO N2 involvement at all\n\n"
            "If no type matches its full set, set confirmed=false and shunt_type=undetermined."
        ),
        expected_output=(
            'JSON only: {"shunt_type": "...", "confirmed": true|false, '
            '"elim_required": true|false, "evidence": "..."}'
        ),
        agent=analyst,
        context=[task1],
    )

    # ── Pre-computation for Task 3 + Task 4 ──────────────────────────────────
    # Q1 corridor fact — numerical truth, injected so model cannot reason around it.
    _in_q1_corridor = 0.04 <= pos_y <= 0.57
    _corridor_fact = (
        f"PROBE IS INSIDE Q1 CORRIDOR (posY {pos_y:.2f} is between 0.04 and 0.57): YES. "
        f"The probe is physically at {current_zone}. Examine HERE — do not route to SFJ/groin."
        if _in_q1_corridor else
        f"PROBE IS INSIDE Q1 CORRIDOR (posY {pos_y:.2f} is between 0.04 and 0.57): NO. "
        f"Probe is at {current_zone} (past all Q1 zones). "
        "If Q1 is still open, route to the nearest unvisited Q1 zone."
    )

    # Zone-level anti-repetition — count how many times each anatomical zone has
    # appeared in recent_guidance, regardless of exact phrasing.
    _zone_keywords = {
        "sfj": "SFJ/groin", "groin": "SFJ/groin",
        "upper thigh": "upper thigh",
        "hunterian": "Hunterian (proximal thigh)", "proximal thigh": "Hunterian (proximal thigh)",
        "dodd": "Dodd (distal thigh)", "distal thigh": "Dodd (distal thigh)",
        "popliteal": "popliteal/SPJ", "spj": "popliteal/SPJ",
        "calf": "calf",
        "ankle": "ankle",
    }
    _zone_hits: Counter = Counter()
    for _g in (recent_guidance or []):
        _g_lo = _g.lower()
        for _kw, _cz in _zone_keywords.items():
            if _kw in _g_lo:
                _zone_hits[_cz] += 1
                break
    _overused_zones = [z for z, c in _zone_hits.items() if c >= 2]
    _overused_zones_str = ", ".join(_overused_zones) if _overused_zones else "none"

    # ── Task 3: Circuit Analyst (sees task1 + task2) ──────────────────────────

    task3 = Task(
        description=(
            "You are the CHIVA Circuit Analyst. Describe the current examination state "
            "and tell the Navigation Planner what the probe should focus on right now.\n\n"
            f"PROBE POSITION: posY={pos_y:.2f} → {current_zone}\n"
            f"{_corridor_fact}\n\n"
            f"Q-STATE:\n{q_state}\n\n"
            f"SCAN HISTORY:\n{history_summary}\n\n"
            "STEP 1 — Identify which examination state we are in:\n"
            "  A: Q1 OPEN — no entry point (EP) clip found yet; surgeon seeking entry\n"
            "  B: Q1 confirmed, Q2 open — entry found; trace GSV trunk distally for RP N2→N1\n"
            "  C: Q1+Q2 confirmed, Q3 open — trunk reflux found; find EP N2→N3 escape\n"
            "  D: Q3 confirmed, Q4 open — escape found; track tributary to RP N3→N1 re-entry\n"
            "  E: Q1-Q4 confirmed, no elimTest — full circuit; elimination test required\n"
            "  F: EP N1→N3 found — SFJ competent; trace N3 tributary to re-entry\n\n"
            "STEP 2 — What should the probe be doing right now?\n\n"
            f"  FOR STATE A:\n"
            f"  • If corridor = YES (probe posY 0.04–0.57): probe is at {current_zone}. "
            f"  The surgeon is already examining a Q1-relevant zone. "
            f"  Describe the specific anatomical structure to find at {current_zone} "
            "  and the surface/direction to locate it (medial, posterior, anterior-medial, etc.).\n"
            "  • If corridor = NO (probe posY > 0.57): the three Q1 candidate zones are "
            "  SFJ/groin (0.04–0.07), Hunterian (proximal thigh) (0.21–0.33), popliteal/SPJ (0.48–0.57). "
            f"  Already routed to recently (do NOT suggest again): {_overused_zones_str}. "
            "  Pick the nearest Q1 candidate NOT in that list. "
            "  If all three are in the list, route to SFJ/groin for re-examination anyway.\n\n"
            "  FOR STATES B–F:\n"
            "  What does the open diagnostic question require at the current probe position? "
            "  Name the specific structure to locate and the direction to find it, "
            "  OR if the probe is in the wrong zone, name the destination zone and why.\n\n"
            "Output ≤80 words. Use ONLY these zone names:\n"
            "SFJ/groin · upper thigh · Hunterian (proximal thigh) · Dodd (distal thigh) · "
            "popliteal/SPJ · calf · ankle"
        ),
        expected_output=(
            "≤80 words: state (A–F), whether probe examines here or moves, "
            "specific anatomical structure + direction if examining here, "
            "or destination zone if moving."
        ),
        agent=circuit,
        context=[task1, task2],
    )

    # ── Task 4: Navigation Planner (sees task3) ───────────────────────────────
    recent_guidance_str = (
        "\n".join(f"  - {g}" for g in (recent_guidance or [])) or "  none"
    )
    _recent_counts = Counter(recent_guidance or [])
    _banned = [g for g, n in _recent_counts.items() if n >= 2]

    task4 = Task(
        description=(
            "You are a senior CHIVA vascular surgeon directing a trainee's duplex probe. "
            "The Circuit Analyst has described the current examination state. "
            "Issue ONE navigation command.\n\n"
            f"Probe at: {current_zone} (posY={pos_y:.2f}) | {len(clips)} clips confirmed\n\n"
            f"Already given — do not repeat:\n{recent_guidance_str}\n\n"
            + (f"'{_recent_counts.most_common(1)[0][0]}' was said "
               f"{_recent_counts.most_common(1)[0][1]}× — find different wording.\n\n"
               if _banned else "") +
            f"OVERUSED ZONES (suggested 2+ times already — route elsewhere): {_overused_zones_str}\n\n"
            "FORBIDDEN TERMS — must NOT appear anywhere in your output: "
            "Q1, Q2, Q3, Q4, corridor, EP, RP, N1, N2, N3, elimTest, elim, "
            "shunt, circuit, 'State A', 'State B', 'State C', 'State D', 'State E', 'State F'.\n\n"
            "A surgical navigation command has TWO components:\n"
            "  MOVEMENT — a direction word: proximally / distally / medially / laterally / "
            "posteriorly / anteriorly\n"
            "  TARGET — ALWAYS includes BOTH the anatomical structure AND its zone location:\n"
            f"    Current zone is {current_zone}. Examples of correct TARGET phrasing:\n"
            "    'GSV in saphenous compartment at upper thigh'\n"
            "    'SFJ junction at groin crease'\n"
            "    'Hunterian perforator on medial proximal thigh'\n"
            "    'SPJ junction in popliteal fossa'\n"
            "    'Dodd perforator on medial distal thigh'\n"
            "    'GSV at medial malleolus (ankle)'\n"
            "    NEVER say just 'GSV' or just 'SPJ junction' without naming where on the leg.\n\n"
            "≤15 words. Start with a movement verb."
        ),
        expected_output=(
            "One navigation command in ≤15 words: movement direction + anatomical structure + zone location."
        ),
        agent=planner,
        context=[task3],
    )

    # ── Task 5: Guidance Specialist (sees task2 + task4 ONLY) ────────────────
    # task3 deliberately excluded — Task 5 must copy task4's movement command,
    # not fall back to the Circuit Analyst's reasoning paragraph.
    task5 = Task(
        description=(
            "Package the Navigation Planner's movement command as JSON. "
            "The Navigation Planner's output is in your context: it is a short movement "
            "command (≤12 words) starting with 'Move'.\n"
            "Output ONLY valid JSON, no markdown:\n"
            '{"guidance": "<the Navigation Planner\'s Move command>", '
            '"action": "move"|"maneuver"|"complete"}\n\n'
            "action:\n"
            "  maneuver — if Shunt Analyst output has elim_required=true\n"
            "  complete  — if Shunt Analyst output has confirmed=true AND the confirmed "
            "shunt_type is NOT already in the surgeon-accepted list: "
            + (", ".join(accepted_shunts) if accepted_shunts else "none (no accepted shunts yet)")
            + "\n"
            "  move      — all other cases, including when the confirmed type is already accepted\n\n"
            "guidance rules:\n"
            "  • Copy the Navigation Planner's 'Move ...' command EXACTLY as written.\n"
            "  • Do NOT copy text from any other agent.\n"
            "  • If action=maneuver, replace guidance with: "
            "'Perform elimination test at current zone'.\n"
            "  • If action=complete, replace guidance with: "
            "'Circuit complete — classification confirmed'."
        ),
        expected_output='{"guidance": "...", "action": "move"|"maneuver"|"complete"}',
        agent=specialist,
        context=[task2, task4],
    )

    # ── Single Crew — one kickoff, agents interact via context ────────────────
    crew = Crew(
        agents=[interpreter, analyst, circuit, planner, specialist],
        tasks=[task1, task2, task3, task4, task5],
        process=Process.sequential,
        verbose=False,
    )

    # Run kickoff in a daemon thread so we can enforce a hard wall-clock limit.
    # The thread keeps running silently if it overruns; it never emits because
    # only the caller (process() in stream.py) calls socketio.emit().
    _result_box:  list = [None]
    _error_box:   list = [None]
    _verbose_box: list = [""]
    _done = threading.Event()

    def _kickoff():
        buf = io.StringIO()
        _saved = sys.stdout
        sys.stdout = buf
        try:
            _result_box[0] = crew.kickoff()
        except Exception as exc:
            _error_box[0] = exc
        finally:
            sys.stdout = _saved
            _verbose_box[0] = buf.getvalue()
            _done.set()

    threading.Thread(target=_kickoff, daemon=True).start()

    _fallback_raw = {"interpret": "", "shunt": "", "circuit": "", "nav": "", "guidance": ""}

    if not _done.wait(timeout=90):
        logger.error("[CrewAI] Crew timed out after 90 s — returning fallback")
        _fallback_raw["guidance"] = "[crew-timeout]"
        return (
            "Continue scanning distally to locate anatomical junction",
            _fallback_raw,
            "move",
            False, "undetermined", "",
        )
    if _error_box[0] is not None:
        logger.error("[CrewAI] Crew kickoff failed: %s", _error_box[0])
        _fallback_raw["guidance"] = f"[crew-error] {_error_box[0]}"
        return (
            "Continue scanning distally to locate anatomical junction",
            _fallback_raw,
            "move",
            False, "undetermined", "",
        )
    result = _result_box[0]

    # ── Extract final output (task5) and intermediate outputs for logging ─────
    raw = result.raw if hasattr(result, "raw") else str(result)

    # tasks_output is a list of TaskOutput objects, one per task
    tasks_out = getattr(result, "tasks_output", [])

    def _t_full(i: int) -> str:
        """Full task output — used for JSON parsing."""
        if i < len(tasks_out):
            o = tasks_out[i]
            return o.raw if hasattr(o, "raw") else str(o)
        return ""

    def _t(i: int) -> str:
        """Truncated task output — used for display only."""
        return _t_full(i)[:80] or "—"

    agents_raw = {
        "interpret": _t_full(0),
        "shunt":     _t_full(1),
        "circuit":   _t_full(2),
        "nav":       _t_full(3),
        "guidance":  raw,
        "verbose":   _verbose_box[0],  # full CrewAI Thought/Action/Observation chain
    }

    # ── Parse JSON from task5 output (guidance) ──────────────────────────────
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

    # ── Parse JSON from task2 output (shunt classification) ──────────────────
    shunt_found    = False
    shunt_type     = "undetermined"
    shunt_evidence = ""

    try:
        t2_raw    = _t_full(1)
        t2_parsed = json.loads(_extract_json_str(t2_raw))
        shunt_found    = bool(t2_parsed.get("confirmed", False))
        shunt_type     = t2_parsed.get("shunt_type", "undetermined")
        shunt_evidence = t2_parsed.get("evidence", "")
    except (json.JSONDecodeError, Exception):
        pass

    return guidance, agents_raw, action, shunt_found, shunt_type, shunt_evidence
