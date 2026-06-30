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
import threading

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

    rejection_ctx = ""
    if rejection_notes:
        rejection_ctx = (
            "SURGEON FEEDBACK — PRIOR CLASSIFICATION REJECTED:\n"
            + "\n".join(f"  • {n}" for n in rejection_notes[-3:])
            + "\n\nRe-evaluate the evidence MORE critically than before. "
            "Do NOT re-confirm the rejected type without a meaningfully different clip set.\n\n"
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
            "You are the second agent. The Clinical Interpreter has assessed the clips. "
            "Using that assessment and the raw clip list below, determine whether a CHIVA "
            "shunt type is confirmed.\n\n"
            f"RAW CONFIRMED CLIPS ({len(clips)} total):\n{clips_text}\n\n"
            "Output ONLY valid JSON (no markdown):\n"
            '  {"shunt_type": "1"|"2A"|"2B"|"2C"|"3"|"1+2"|"4"|"5"|"6"|"undetermined",\n'
            '   "confirmed": true|false,\n'
            '   "evidence": "<one sentence: clip pattern that confirms or why unconfirmed>"}\n\n'
            '"confirmed" is true ONLY when the FULL minimum clip set is present:\n'
            "  Type 1   → EP N1→N2 (SFJ or Hunterian) + RP N2→N1; NO N3 clips at all\n"
            "  Type 2A  → EP N2→N3 + RP N3→N1; NO EP N1→N2; NO EP N2→N2; NO RP N2→N1\n"
            "             (SFJ competent; GSV escapes into tributary; tributary re-enters deep)\n"
            "  Type 2B  → EP N2→N2 + RP N3→N1; NO EP N1→N2; NO RP N2→N1\n"
            "             (perforator/SPJ feeds system; tributary re-enters deep; no trunk reflux)\n"
            "  Type 2C  → (EP N2→N2 or EP N2→N3) + RP N3 + RP N2→N1; NO EP N1→N2\n"
            "             (perforator entry + secondary GSV trunk reflux back to N1; SFJ still competent)\n"
            "  Type 3   → EP N1→N2 + EP N2→N3 + RP N3→N1 + RP N2→N1 + elimTest=No Reflux\n"
            "             (elimination test is MANDATORY — 3 clips without RP N2→N1 is NOT confirmed Type 3;\n"
            "              trunk reflux must be confirmed via RP N2→N1, then elim test performed)\n"
            "  Type 1+2 → EP N1→N2 + EP N2→N3 + RP N3→N1 + RP N2→N1 + elimTest=Reflux\n"
            "             (same clips as Type 3 but elim test shows trunk reflux is independent, not conducted)\n"
            "  Type 4   → EP N1→N3 + RP N2→N1 (GSV trunk carries return blood back to N1; trunk REFLUXES)\n"
            "             optional intermediate: EP N1→N3 + RP N3→N2 + RP N2→N1\n"
            "  Type 5   → EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1; NO RP N2→N1\n"
            "             (blood loops N1→N3→N2→N3→N1; GSV loops but never directly drains to N1)\n"
            "  Type 6   → EP N1→N3 + RP N3→N1; NO N2 clips whatsoever\n"
            "             (perforator→tributary→perforator back to deep; GSV completely bypassed)\n\n"
            "KEY DIFFERENTIATOR — all three N1→N3-entry types:\n"
            "  Type 4: RP N2→N1 PRESENT (trunk refluxes directly to deep)\n"
            "  Type 5: EP N2→N3 PRESENT, NO RP N2→N1 (trunk loops back to N3 before returning)\n"
            "  Type 6: NO N2 involvement at all (tributary returns straight to N1 via perforator)\n\n"
            "If no type matches its full set, set confirmed=false and shunt_type=undetermined."
        ),
        expected_output='JSON only: {"shunt_type": "...", "confirmed": true|false, "evidence": "..."}',
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
            "FORBIDDEN words in guidance text (never use these):\n"
            "  Clinical terms: EP, RP, N1, N2, N3, reflux, Q1, Q2, Q3, Q4,\n"
            "    confirmed, findings, diagnostic, shunt, Given, Since, As the, Currently\n"
            "  Maneuver names: Paranà, Parana, Valsalva, squeezing, compression, maneuver\n"
            "  Paranà and Valsalva are maneuver TECHNIQUES, NOT anatomical destinations —\n"
            "  NEVER write 'Move ... to Paranà' or 'Move ... to Valsalva'.\n\n"
            "guidance format rules:\n"
            "  - action='move'     → probe navigation: 'Move probe <direction> to <anatomical target>'\n"
            "  - action='maneuver' → position instruction: 'Hold probe at <target> and compress'\n"
            "  - action='complete' → 'Circuit complete — all zones confirmed'\n"
            "  - MUST contain one direction word: distally / proximally / medially /\n"
            "    posteriorly / laterally / transversely / deeper\n"
            "  - anatomical targets: SFJ, GSV, SPJ, SSV, popliteal, mid-thigh, calf, ankle, perforator"
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

    # Run kickoff in a daemon thread so we can enforce a hard wall-clock limit.
    # The thread keeps running silently if it overruns; it never emits because
    # only the caller (process() in stream.py) calls socketio.emit().
    _result_box: list = [None]
    _error_box:  list = [None]
    _done = threading.Event()

    def _kickoff():
        try:
            _result_box[0] = crew.kickoff()
        except Exception as exc:
            _error_box[0] = exc
        finally:
            _done.set()

    threading.Thread(target=_kickoff, daemon=True).start()

    if not _done.wait(timeout=90):
        logger.error("[CrewAI] Crew timed out after 90 s — returning fallback")
        return (
            "Continue scanning distally to locate anatomical junction",
            "[crew-timeout]",
            "move",
            False, "undetermined", "",
        )
    if _error_box[0] is not None:
        logger.error("[CrewAI] Crew kickoff failed: %s", _error_box[0])
        return (
            "Continue scanning distally to locate anatomical junction",
            f"[crew-error] {_error_box[0]}",
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

    combined_raw = (
        f"[interpret] {_t(0)} | "
        f"[shunt] {_t(1)} | "
        f"[circuit] {_t(2)} | "
        f"[nav] {_t(3)} | "
        f"[guidance] {raw[:80]}"
    )

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
        t2_raw = _t_full(1)
        t2_parsed = json.loads(_extract_json_str(t2_raw))
        shunt_found    = bool(t2_parsed.get("confirmed", False))
        shunt_type     = t2_parsed.get("shunt_type", "undetermined")
        shunt_evidence = t2_parsed.get("evidence", "")
    except (json.JSONDecodeError, Exception):
        pass

    # ── Deterministic gate (bidirectional) ────────────────────────────────────
    # Blocks LLM false-positives AND fills in LLM false-negatives.
    # Also drives the action gate below.
    def _has(flow: str, fT: str, tT: str) -> bool:
        return any(
            c.get("flow") == flow and c.get("from_type") == fT and c.get("to_type") == tT
            for c in clips
        )

    ep_n1_n2       = _has("EP", "N1", "N2")
    ep_n2_n2       = _has("EP", "N2", "N2")
    ep_n2_n3       = _has("EP", "N2", "N3")
    ep_n1_n3       = _has("EP", "N1", "N3")
    rp_n2_n1       = _has("RP", "N2", "N1")
    rp_n3_n1       = _has("RP", "N3", "N1")
    rp_n3_n2       = _has("RP", "N3", "N2")
    elim_no_reflux = any(c.get("elimination_test", "") == "No Reflux" for c in clips)
    elim_reflux    = any(c.get("elimination_test", "") == "Reflux"    for c in clips)
    has_elim       = elim_no_reflux or elim_reflux

    # Elimination test trigger: requires SFJ/SPJ entry (ep_n1_n2) to be meaningful.
    # Without ep_n1_n2, ep_n2_n3 + rp_n3_n1 + rp_n2_n1 is Type 2C, not an ambiguous trunk.
    elim_trigger = ep_n1_n2 and ep_n2_n3 and rp_n3_n1 and rp_n2_n1 and not has_elim

    # Forward-override checks (priority order, most restrictive first).
    # Type 2A intentionally omitted: ep_n2_n3 + rp_n3_n1 is a valid intermediate state
    # for Type 2C and Type 3/1+2, so premature auto-confirmation would be wrong.
    _forward_checks: list[tuple[str, bool]] = [
        ("3",   ep_n1_n2 and ep_n2_n3 and rp_n3_n1 and rp_n2_n1 and elim_no_reflux),
        ("1+2", ep_n1_n2 and ep_n2_n3 and rp_n3_n1 and rp_n2_n1 and elim_reflux),
        ("1",   ep_n1_n2 and rp_n2_n1 and not ep_n2_n3 and not rp_n3_n1),
        ("5",   ep_n1_n3 and rp_n3_n2 and ep_n2_n3 and rp_n3_n1 and not rp_n2_n1),
        ("4",   ep_n1_n3 and rp_n2_n1),
        ("6",   ep_n1_n3 and rp_n3_n1 and not ep_n2_n2 and not ep_n2_n3
                and not rp_n2_n1 and not rp_n3_n2),
        ("2C",  not ep_n1_n2 and (ep_n2_n2 or ep_n2_n3)
                and (rp_n3_n1 or rp_n3_n2) and rp_n2_n1),
        ("2B",  not ep_n1_n2 and ep_n2_n2 and rp_n3_n1 and not rp_n2_n1),
    ]

    # Minimum-set map for LLM false-positive blocking (all types, including 2A)
    _all_min_ok: dict[str, bool] = {
        "1":   ep_n1_n2 and rp_n2_n1,
        "2A":  ep_n2_n3 and rp_n3_n1,
        "2B":  ep_n2_n2 and rp_n3_n1,
        "2C":  (ep_n2_n2 or ep_n2_n3) and (rp_n3_n1 or rp_n3_n2) and rp_n2_n1,
        "3":   ep_n1_n2 and ep_n2_n3 and rp_n3_n1 and rp_n2_n1 and has_elim,
        "1+2": ep_n1_n2 and ep_n2_n3 and rp_n3_n1 and rp_n2_n1 and has_elim,
        "4":   ep_n1_n3 and rp_n2_n1,
        "5":   ep_n1_n3 and rp_n3_n2 and ep_n2_n3 and rp_n3_n1,
        "6":   ep_n1_n3 and rp_n3_n1,
    }

    det_type      = "undetermined"
    det_confirmed = False
    for stype, ok in _forward_checks:
        if ok:
            det_type      = stype
            det_confirmed = True
            break

    if det_confirmed:
        shunt_found    = True
        shunt_type     = det_type
        if not shunt_evidence:
            shunt_evidence = f"Minimum clip set for Type {det_type} complete."
    elif shunt_found:
        # LLM said confirmed but minimum clips are absent → block false positive
        if not _all_min_ok.get(shunt_type, True):
            shunt_found = False

    # ── Action gate ───────────────────────────────────────────────────────────
    if det_confirmed:
        # Full circuit confirmed deterministically — always override guidance
        # so the text contains the required keywords even when LLM returned
        # action=complete with weak phrasing.
        guidance = "Circuit complete — classification confirmed, all zones mapped"
        action = "complete"
    elif action == "maneuver" and not elim_trigger:
        # Spurious maneuver: elimination test condition not actually met
        action = "move"
    elif elim_trigger and action != "maneuver":
        # Elimination test is pending but LLM missed it
        action   = "maneuver"
        guidance = "Hold probe at escape site and compress tributary — record Doppler response"

    return guidance, combined_raw, action, shunt_found, shunt_type, shunt_evidence
