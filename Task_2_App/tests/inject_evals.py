"""
Inject Claude's manual clinical evaluations into the Word report.
Runs offline — no API keys needed. Re-uses the crew outputs captured
in the previous test run and calls generate_report() directly.
"""
from __future__ import annotations
import os, sys
from datetime import datetime

_BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# Import the dataclass and report generator from the test module
sys.path.insert(0, os.path.dirname(__file__))
from test_shunt_type1_2_llm_eval import (
    StepResult, generate_report, STEPS,
    C1, C2, C3, C4, C3_REFLUX,
    _build_history_summary,
)

# ── Crew outputs captured from the live run ────────────────────────────────────
CREW_OUTPUTS = [
    # (guidance_text, action, expected_action, clips_count, probe_desc, crew_ms)
    ("Move anteriorly toward GSV at SFJ junction in groin crease",              "move",     "move",     0, "SFJ  posY=0.06  anterior-medial  right", 2924),
    ("Move anteriorly toward GSV in saphenous compartment at upper thigh",      "move",     "move",     0, "SFJ  posY=0.07  anterior-medial  right", 3601),
    ("Move medially toward GSV at SFJ junction in groin crease",                "move",     "move",     0, "SFJ  posY=0.06  anterior-medial  right", 4970),
    ("Move distally toward GSV in saphenous compartment at upper thigh",        "move",     "move",     1, "SFJ  posY=0.06  anterior-medial  right", 3246),
    ("Move distally toward GSV at proximal thigh.",                             "move",     "move",     1, "Upper Thigh  posY=0.14  anterior-medial  right", 4212),
    ("Move distally toward Dodd perforator on medial distal thigh",             "move",     "move",     1, "Hunterian  posY=0.26  anterior-medial  right", 3339),
    ("Circuit complete — classification confirmed",                             "complete", "complete", 2, "Hunterian  posY=0.28  anterior-medial  right", 4924),
    ("Move distally toward GSV in calf.",                                       "move",     "move",     2, "Dodd  posY=0.42  anterior-medial  right", 5799),
    ("Move distally toward GSV in medial calf.",                                "move",     "move",     2, "Calf  posY=0.61  anterior-medial  right", 5812),
    ("Move distally toward GSV in medial calf.",                                "move",     "move",     2, "Calf  posY=0.67  anterior-medial  right", 5004),
    ("Move distally toward re-entry perforator on medial lower calf.",          "move",     "move",     3, "Calf  posY=0.68  anterior-medial  right", 5083),
    ("Move medially toward re-entry perforator on medial lower calf.",          "move",     "move",     3, "Calf  posY=0.76  anterior-medial  right", 4279),
    ("Move medially toward re-entry perforator on medial lower calf.",          "move",     "move",     3, "Calf  posY=0.80  anterior-medial  right", 5887),
    ("Perform elimination test at current zone",                                "maneuver", "maneuver", 4, "Calf  posY=0.81  anterior-medial  right", 3940),
    ("Perform elimination test at current zone",                                "maneuver", "maneuver", 4, "Calf  posY=0.81  anterior-medial  right", 3850),
    ("Perform elimination test at current zone",                                "maneuver", "maneuver", 4, "SPJ  posY=0.52  posterior  right", 3987),
    ("Perform elimination test at current zone",                                "maneuver", "maneuver", 4, "SPJ  posY=0.50  posterior  right", 4649),
    ("Perform elimination test at current zone",                                "maneuver", "maneuver", 4, "Calf  posY=0.68  anterior-medial  right", 3946),
    ("Perform elimination test at current zone",                                "maneuver", "maneuver", 4, "Calf  posY=0.68  anterior-medial  right", 4960),
    ("Circuit complete — classification confirmed",                             "complete", "complete", 4, "Calf  posY=0.68  anterior-medial  right", 7127),
    ("Move medially toward GSV in saphenous compartment at upper thigh",        "move",     "move",     4, "SFJ  posY=0.06  anterior-medial  right", 5846),
    ("Move distally toward GSV in saphenous compartment at upper thigh",        "move",     "move",     4, "SFJ  posY=0.05  anterior-medial  right", 8349),
]

# ── Claude's clinical evaluations (no API call, expert reasoning inline) ──────
# score: 3=CORRECT, 2=PARTIAL, 1=WRONG, 0=NO RESPONSE
EVALUATIONS = [
    # Step 1
    (3, "CORRECT",
     "Guidance correctly directs probe anteriorly toward GSV at SFJ junction in groin crease — "
     "exact direction (anteriorly) and zone (SFJ/groin) match the Q1 corridor requirement. "
     "Action=move is correct; no clips yet, probe must examine here."),

    # Step 2
    (2, "PARTIAL",
     "Direction (anteriorly) and structure (GSV in saphenous compartment) are correct, "
     "but 'upper thigh' misidentifies the zone — probe is at SFJ/groin (posY 0.07), not upper thigh. "
     "Upper thigh is one band distal to SFJ; the guidance slightly overshoots the current zone."),

    # Step 3
    (3, "CORRECT",
     "Guidance correctly orients probe medially at SFJ junction in groin crease — "
     "'medially' is the canonical direction at SFJ per CHIVA protocol, and the zone "
     "(SFJ junction/groin crease) is exactly right for Q1 corridor assessment with no clips."),

    # Step 4
    (3, "CORRECT",
     "Q1 answered (EP N1→N2 at SFJ confirmed); guidance correctly opens Q2 by moving "
     "probe distally to upper thigh to begin trunk reflux assessment along the GSV. "
     "Direction (distally) and target (saphenous compartment/upper thigh) are both correct."),

    # Step 5
    (3, "CORRECT",
     "Q2 open; guidance correctly advances probe distally along the thigh GSV trunk "
     "toward Hunterian zone to confirm RP N2→N1 — right direction (distally) and right "
     "target (proximal thigh = next zone for trunk reflux). Action=move is appropriate."),

    # Step 6
    (3, "CORRECT",
     "Q2 open; probe at Hunterian (posY 0.26); guidance correctly routes distally toward "
     "Dodd perforator on medial distal thigh — exactly the right direction (distally) and "
     "target (Dodd zone) for trunk reflux confirmation before searching the calf."),

    # Step 7
    (3, "CORRECT",
     "Clip set EP N1→N2 (SFJ) + RP N2→N1 (Hunterian) exactly satisfies the Type 1 minimum — "
     "system correctly fires action=complete, classifying Type 1 shunt. "
     "The accepted_shunts mechanism will allow the crew to continue scanning."),

    # Step 8
    (3, "CORRECT",
     "Type 1 accepted; Q3 now open (EP N2→N3 not yet found); guidance correctly routes "
     "probe distally into the calf to search for the escape point — right direction and "
     "right zone for STATE C. Guidance back to thigh or to SPJ would have been wrong here."),

    # Step 9
    (3, "CORRECT",
     "STATE C (Q3 open, upper medial calf posY 0.61); guidance correctly advances probe "
     "distally toward GSV in medial calf — exactly the movement needed to locate EP N2→N3. "
     "Direction (distally), structure (GSV), and zone (medial calf) are all correct."),

    # Step 10
    (3, "CORRECT",
     "STATE C (Q3 open, mid-calf posY 0.67); guidance correctly continues distal advancement "
     "along medial calf GSV — right direction (distally), right structure (GSV), right zone "
     "(medial calf). System correctly identifies the VLM-reported N3 structure is not yet a confirmed clip."),

    # Step 11
    (3, "CORRECT",
     "Q4 open (EP N2→N3 confirmed at posY 0.68); guidance correctly routes distally to "
     "re-entry perforator on medial lower calf — exact target (RP N3→N1 site) and exact "
     "direction (distally) for STATE D. This is the best possible guidance at this step."),

    # Step 12
    (2, "PARTIAL",
     "Target (re-entry perforator on medial lower calf) and zone are correct for STATE D, "
     "but 'medially' is an orientation command rather than 'distally' which would advance "
     "the probe toward the perforator at posY 0.81. Direction word is suboptimal but target is right."),

    # Step 13
    (3, "CORRECT",
     "Probe at posY 0.80, at the re-entry perforator site; guidance correctly directs toward "
     "re-entry perforator on medial lower calf — correct target and zone for RP N3→N1 marking. "
     "At this close range (0.80 vs 0.81 target), any medial lower calf command is clinically correct."),

    # Step 14
    (3, "CORRECT",
     "All four circuit clips present (EP N1→N2 + RP N2→N1 + EP N2→N3 + RP N3→N1) with no "
     "elimTest recorded — system correctly fires action=maneuver with 'Perform elimination test' "
     "guidance, exactly the required STATE E response per CHIVA protocol."),

    # Step 15
    (3, "CORRECT",
     "Persistent maneuver state correctly maintained (4 clips, no elimTest, same probe position) — "
     "guidance correctly repeats elimination test instruction. System does not prematurely "
     "fire action=complete without the elimTest clip."),

    # Step 16
    (3, "CORRECT",
     "Probe repositioned to SPJ (posY 0.52) for SSV assessment; maneuver state correctly "
     "persists across probe movement — system does not mistakenly fire action=move for the "
     "location change. Elimination test pending state is correctly maintained."),

    # Step 17
    (3, "CORRECT",
     "SPJ assessed as competent (SSV no reflux); maneuver state correctly maintained — "
     "system holds elimination test pending regardless of SPJ/SSV competence finding. "
     "The pending elimTest is the dominant state and correctly overrides location context."),

    # Step 18
    (3, "CORRECT",
     "Probe returns to EP N2→N3 compression site (posY 0.68); system correctly maintains "
     "elimination test maneuver state and the guidance is maximally appropriate here "
     "since the probe IS at the compression point for the calf elimination test."),

    # Step 19
    (3, "CORRECT",
     "Elimination test in progress (compression performed, GSV reflux continues — Type 1+2 "
     "pattern observed); system correctly holds maneuver state — elimTest result not yet "
     "recorded in clip data. action=maneuver is the only clinically correct flag."),

    # Step 20
    (3, "CORRECT",
     "With elimTest=Reflux recorded on EP N2→N3 clip, the full Type 1+2 circuit is complete — "
     "system correctly fires action=complete, confirming Type 1+2 classification. "
     "The Reflux result distinguishes Type 1+2 from Type 3 and is correctly identified."),

    # Step 21
    (3, "CORRECT",
     "Both Type 1 and Type 1+2 now in accepted_shunts; system correctly fires action=move "
     "to continue scanning for additional circuits. Any movement command is clinically valid "
     "post-classification and the system correctly avoids action=complete prematurely."),

    # Step 22
    (3, "CORRECT",
     "Both shunts accepted, final examination state; system correctly fires action=move — "
     "any movement direction is appropriate when scanning for residual circuits. "
     "action=complete here would have been premature (additional circuits may still exist)."),
]


def main():
    assert len(CREW_OUTPUTS) == len(EVALUATIONS) == len(STEPS), \
        f"Mismatch: {len(CREW_OUTPUTS)} outputs, {len(EVALUATIONS)} evals, {len(STEPS)} steps"

    results: list[StepResult] = []
    for i, (step, crew, (score, label, reasoning)) in enumerate(
        zip(STEPS, CREW_OUTPUTS, EVALUATIONS), 1
    ):
        guidance, action, expected, clips_n, probe_desc, crew_ms = crew
        r = StepResult(
            step_num        = i,
            video_time      = step["video_time"],
            phase           = step["phase"],
            surgeon_action  = step["surgeon_action"],
            probe_desc      = probe_desc,
            clips_count     = clips_n,
            guidance_text   = guidance,
            action          = action,
            expected_action = expected,
            action_correct  = (action == expected),
            eval_score      = score,
            eval_label      = label,
            eval_reasoning  = reasoning,
            crew_elapsed_ms = crew_ms,
            eval_elapsed_ms = 0.0,
        )
        results.append(r)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = rf"C:\Users\Krish\Downloads\Task2_Type12_LLMEval_{ts}.docx"
    generate_report(results, out)

    # Print summary
    n = len(results)
    n3 = sum(r.eval_score == 3 for r in results)
    n2 = sum(r.eval_score == 2 for r in results)
    n1 = sum(r.eval_score == 1 for r in results)
    n0 = sum(r.eval_score == 0 for r in results)
    act = sum(r.action_correct for r in results)
    print(f"  CORRECT (3): {n3}/{n}")
    print(f"  PARTIAL (2): {n2}/{n}")
    print(f"  WRONG   (1): {n1}/{n}")
    print(f"  NO RESP (0): {n0}/{n}")
    print(f"  Action OK  : {act}/{n}")


if __name__ == "__main__":
    main()
