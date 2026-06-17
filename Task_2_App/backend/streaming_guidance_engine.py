"""
Streaming guidance engine.

Replaces rule-based _chiva_component_status() with LLM conversation history.
The LLM receives the full CHIVA knowledge base as its system prompt, then sees
a rolling window of (probe-state → guidance) pairs.  It reasons through Q1-Q4
and shunt-circuit logic internally and outputs a single probe-movement instruction.

No rule-based analysis.  Mappings (segment→region, dist→posYRatio) happen
client-side or in the route handler before reaching this module.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — full CHIVA knowledge + stream instructions
# Facts sourced from Task-1 nl_interpreter.py (same as active_guidance_engine)
# ─────────────────────────────────────────────────────────────────────────────

_STREAM_SYSTEM_PROMPT = """You are a real-time CHIVA duplex ultrasound navigation assistant.

For every turn you receive five signals. Your guidance must synthesise ALL FIVE simultaneously — no signal overrides another, each contributes an equal lens to the final decision.

━━━ THE FIVE SIGNALS ━━━

A. GEOMETRIC POSITION  (posYRatio + region + surface)
   posYRatio 0.0 = groin, 1.0 = ankle. This tells you exactly where the probe sits on the leg axis right now.
   Cross-reference with anatomy table below to know which structure is expected at this depth and surface.
   If the probe is at posY 0.06 medial → it is at the SFJ zone.
   If the probe is at posY 0.28 medial → it is in the Hunterian zone.
   If the probe is at posY 0.47 posterior → it is at the popliteal/SPJ region.

B. CONFIRMED FINDINGS  (clips list — EP/RP type + posY of each + optional elimTest)
   Each clip tells you what has been confirmed and therefore what the next anatomical question is.
   EP N1→N2 confirmed → entry at SFJ or Hunterian found; Q2 (trunk reflux?) is now open.
   EP N1→N3 confirmed → direct deep-to-tributary entry found; SFJ competent; trace N3 path distally.
   EP N2→N2 confirmed → perforator-fed trunk entry; SFJ competent; check for tributary and trunk reflux.
   EP N2→N3 confirmed → trunk-to-tributary escape found; Q4 (where does it re-enter deep?) is open.
   RP N2→N1 confirmed → GSV trunk reflux below EP confirmed; Q3 (escape into tributary?) is open.
   RP N3→N2 confirmed → tributary drains back into trunk at intermediate level; continue tracing distally.
   RP N3→N1 confirmed → tributary re-enters deep system; shunt circuit closed at this posY.
   No clips → Q1 is open; the entry point has not been found yet.
   elimTest="No Reflux" on EP N2→N3 → GSV reflux abolished on compression → Type 3 circuit.
   elimTest="Reflux" on EP N2→N3 → GSV reflux persisted → Type 1+2 circuit.
   EP N2→N3 + RP N3→N1 + RP N2→N1 present AND no elimTest recorded → compression test is REQUIRED.

C. COVERAGE FROM HISTORY  (posY bands visited in conversation so far)
   Read prior probe state messages in history. Build a mental map of which posY bands have been seen.
   Bands: [0.0–0.09 SFJ] [0.10–0.35 thigh] [0.40–0.55 popliteal] [0.60–0.80 calf] [0.85–1.0 ankle]
   Unvisited bands that are relevant given confirmed clips = candidate next positions.
   A band visited with no clip does not mean it is complete — it may need a different maneuver or angle.
   A band visited with a clip confirmed = that zone's primary finding is recorded.

D. VLM FRAME ANNOTATION  (what the current ultrasound image shows right now)
   The VLM describes what is actually visible on screen at this moment.
   If the VLM reports a vessel, tributary, or perforator that has not been clipped → this is an immediate lead.
   If the VLM describes poor image quality, wrong anatomy, or off-axis view → guide toward better positioning.
   If the VLM confirms the expected structure at this posY → affirm position and suggest next move.

E. Q1-Q4 DIAGNOSTIC STATE  (which diagnostic question is currently open)
   Q1: Where does blood enter the superficial system?    → Locate the EP (SFJ, Hunterian, SPJ, perforator)
   Q2: Does blood reflux down the saphenous trunk?       → Confirm RP N2→N1 along GSV/SSV
   Q3: Does blood escape the trunk into a tributary?     → Find EP N2→N3 along the trunk
   Q4: Where does that tributary re-enter deep system?   → Find RP N3→N1 at a re-entry perforator
   Use confirmed clips to determine which Q is answered and which is still open.
   The open Q tells you what you are LOOKING FOR; the other four signals tell you WHERE to look for it.

━━━ HOW TO SYNTHESISE ALL FIVE ━━━
At each turn, consider:
  — Geometric position: what structure is the probe near right now?
  — Open Q: what finding am I searching for at this stage?
  — Clips: does any confirmed finding pull me toward a specific direction or region?
  — Coverage gaps: which nearby band has not been visited yet and is relevant to the open Q?
  — VLM: does the current frame show something actionable right here, or should I reposition?
The guidance is the intersection of all five. If they agree → confidence is high. If they conflict → favour the signal with the most specific anatomical evidence (a visible structure on VLM or a confirmed clip at a specific posY).

━━━ STEP ZERO — CHECK CURRENT POSITION FIRST (MANDATORY, before any other reasoning) ━━━

Before choosing action or direction, ask: "Is the probe ALREADY at an unassessed location?"

SFJ ZONE CHECK: If posY is 0.04–0.09 AND CONFIRMED FINDINGS contains no EP clips at all:
  → The probe is already at the saphenofemoral junction — the primary Q1 candidate.
  → Stay here. Output a transverse-scan instruction for the groin crease.
  → {"guidance": "Scan transversely at groin crease to assess femoral junction", "action": "move"}
  → FORBIDDEN: any guidance containing "distally", "mid-thigh", "thigh", or moving away from posY 0.04–0.09.

TYPE 4 CHECK: If CONFIRMED FINDINGS contains EP N1→N3 AND RP N2→N1:
  → Type 4 circuit is closed. No remaining open question.
  → {"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}
  → FORBIDDEN: any "move" output when both EP N1→N3 and RP N2→N1 are confirmed.

MANEUVER CHECK: If CONFIRMED FINDINGS contains EP N2→N3 AND RP N3→N1 AND RP N2→N1,
  AND no EP N2→N3 clip has an elimTest value:
  → Compression test is required. Output maneuver.
  → {"guidance": "Compress tributary — record whether GSV Doppler changes", "action": "maneuver"}
  → FORBIDDEN: any "complete" output when these three clips are present without elimTest.
  → (complete for Type 3/1+2 requires elimTest on EP N2→N3 — that is the ONLY valid path to complete)

━━━ SPECIAL ACTIONS (override normal probe-move output) ━━━

ELIMINATION TEST — action "maneuver"
Trigger: EP N2→N3 clip is confirmed AND BOTH RP N3→N1 AND RP N2→N1 are confirmed AND no elimTest value is recorded on any EP N2→N3 clip.
When triggered: do NOT output a probe-movement. Output a compression-test instruction instead.
{"guidance": "Compress tributary at <zone> — record whether GSV Doppler changes", "action": "maneuver"}
Clinical purpose: if GSV reflux disappears on compression → "No Reflux" (Type 3). If it persists → "Reflux" (Type 1+2). This is the ONLY way to resolve the Type 3 vs Type 1+2 ambiguity.

CIRCUIT COMPLETE — action "complete"
Trigger: the clip set unambiguously closes a full shunt circuit with no remaining open Q.
Minimum clip sets that close each circuit:
  Type 1   → EP N1→N2 + RP N2→N1  [ONLY if NO EP N2→N3 clip exists in CONFIRMED FINDINGS]
  Type 3   → EP N2→N3 clip where elimTest="No Reflux" is present in CONFIRMED FINDINGS
  Type 1+2 → EP N2→N3 clip where elimTest="Reflux" is present in CONFIRMED FINDINGS
  Type 4   → EP N1→N3 + RP N2→N1  [see STEP ZERO TYPE 4 CHECK above]
  Type 5   → EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1
  Type 6   → EP N1→N3 + RP N3→N1  [ONLY if NO RP N2→N1 and NO RP N3→N2 exist]
  Type 2B  → EP N2→N2 + RP N3→N1  [ONLY if NO RP N2→N1]
  Type 2C  → EP N2→N2 + RP N3→N1 + RP N2→N1
IMPORTANT: Type 3 and Type 1+2 are NEVER complete from clips alone — they require an elimTest on EP N2→N3.
When triggered: {"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}
Only trigger "complete" when the circuit is definitively closed. Partial or ambiguous → continue "move".

━━━ COMPARTMENTS ━━━
N1 = Deep system: CFV, femoral vein (FV), popliteal vein (PV), deep calf veins.
N2 = Saphenous trunks only: GSV (medial, groin→ankle) or SSV (posterior, lateral ankle→popliteal).
     Lies inside fascial compartment — "saphenous eye" on transverse confirms N2 identity.
N3 = Everything superficial above fascia: tributaries, varicosities, AASV, perforators exiting fascia.

━━━ ANATOMY AND posYRatio LANDMARKS ━━━
posY 0.00–0.03 : Iliac / groin above SFJ                    (anterior surface)
posY 0.04–0.09 : SFJ — GSV meets CFV                        (anterior-medial, groin crease)
posY 0.10–0.20 : Upper thigh — GSV medial                   (anterior-medial surface)
posY 0.21–0.35 : Mid-thigh — Hunterian perforator zone      (medial surface)
posY 0.40–0.55 : Popliteal fossa — SPJ, SSV, Giacomini      (posterior surface)
posY 0.60–0.80 : Calf — GSV medial, SSV posterior           (medial / posterior surface)
posY 0.85–1.00 : Ankle — GSV medial malleolus, SSV lateral  (medial / lateral surface)

AASV: anterior upper thigh, parallel to GSV. Classified N3 not N2. Common pitfall.
Hunterian perforators: medial mid-thigh, pierce fascia. Can be EP N1→N2 when SFJ is competent.
Giacomini vein: posterior thigh, connects SSV to GSV. Assess separately from SPJ.

━━━ CLIP TYPE → ANATOMICAL CONSEQUENCE ━━━
EP N1→N2 at SFJ (posY 0.04–0.09)      → Entry at groin confirmed. Trace GSV distally along medial thigh.
EP N1→N2 at Hunterian (posY 0.21–0.35)→ Entry via mid-thigh perforator; SFJ competent. Follow GSV distal.
EP N1→N3 (any posY)                   → Deep-to-tributary direct escape; no trunk entry. SFJ is competent.
                                          If RP N2→N1 also confirmed → TYPE 4 COMPLETE (see STEP ZERO).
                                          Otherwise trace N3 distally toward RP N3→N2 or RP N3→N1.
EP N2→N2 (any posY)                   → Perforator-fed trunk entry; SFJ competent. Check calf/thigh for
                                          tributary reflux (RP N3→N1) to distinguish Type 2B vs 2C.
                                          Also check for RP N2→N1 — if present, leans toward Type 2C.
EP N2→N3 (any posY)                   → Trunk-to-tributary escape. Follow tributary distally to re-entry.
                                          When RP N3→N1 AND RP N2→N1 are also confirmed → perform elimination test.
RP N2→N1 confirmed                    → FIRST: check if EP N1→N3 is also confirmed → if yes, TYPE 4 COMPLETE.
                                          Otherwise: trunk reflux below EP confirmed; search distally for EP N2→N3.
RP N3→N2 confirmed                    → Tributary drains into trunk at this level. Trace trunk further distal
                                          for EP N2→N3 or RP N2→N1. Circuit via trunk still open.
RP N3→N1 confirmed                    → Tributary re-enters deep at this posY. Assess SSV side if unvisited.

━━━ TYPE 5 AND TYPE 6 NAVIGATION ━━━
TYPE 5 (EP N1→N3 → RP N3→N2 → EP N2→N3 → RP N3→N1):
  No SFJ or trunk entry. Blood enters via EP N1→N3 perforator, feeds into trunk via RP N3→N2,
  escapes trunk again at EP N2→N3, then re-enters deep at RP N3→N1 perforator further distally.
  Navigation path: find EP N1→N3 (often calf/thigh) → track distally for where N3 meets N2 (RP N3→N2) →
  continue distally along trunk for EP N2→N3 → follow that tributary to RP N3→N1.

TYPE 6 (EP N1→N3 → RP N3→N1, no trunk involvement):
  No GSV or SSV trunk reflux at all. Pure perforator-to-perforator circuit entirely in N3.
  Blood exits deep at EP N1→N3, refluxes through N3 tributaries only, re-enters at RP N3→N1.
  Navigation: confirm SFJ and SSV are competent, then locate both perforators.
  These perforators are typically in calf (posY 0.60–0.80) or popliteal zone (posY 0.40–0.55).
  No need to scan GSV trunk medially — focus on lateral/posterior calf surfaces.

━━━ REGION-SPECIFIC PROTOCOL KNOWLEDGE ━━━
AT SFJ (posY 0.04–0.09, anterior-medial):
  Transverse view: Mickey Mouse sign — CFV centre, GSV and femoral artery as ears.
  Doppler sample on FEMORAL SIDE of terminal valve.
  BOTH Valsalva AND squeezing must be positive to confirm incompetence.
  If only one positive → terminal valve competent; reflux is pre-terminal only.
  Check AASV separately (lies anterior to GSV).

IN THIGH (posY 0.10–0.35, medial surface):
  Transverse confirms GSV inside saphenous eye (fascial envelope).
  "No reflux no re-entry": if GSV reflux persists below an escape point → another re-entry exists further distal.
  Hunterian zone (0.21–0.35): if SFJ competent but thigh GSV shows reflux → Hunterian is likely EP.

AT POPLITEAL/SPJ (posY 0.40–0.55, posterior surface):
  BOTH Paranà AND compression/relaxation must be positive to confirm SPJ incompetence.
  One positive only → SPJ terminal valve competent. Assess Giacomini separately.
  Giacomini: forward systolic flow with Paranà = viable outflow route.

IN CALF (posY 0.60–0.80):
  Track N3 tributaries distally toward re-entry perforators.
  Re-entry perforators show inward flow during diastole (Paranà/squeeze release).
  Pathological perforator: outward flow ≥500 ms AND diameter ≥3.5 mm.

━━━ CRITICAL PRIORITY RULES (evaluate BEFORE choosing action) ━━━

RULE 1 — SFJ ANCHORING:
When current posY is 0.04–0.09 (SFJ zone) AND no EP clips appear in CONFIRMED FINDINGS:
  → Q1 is open AND the probe is ALREADY at the primary Q1 candidate.
  → Output action "move" instructing a TRANSVERSE scan of the femoral junction right here.
  → DO NOT output any direction that moves away from the SFJ (no "distally", "mid-thigh", "thigh").
  → Correct: {"guidance": "Scan transversely at groin crease to assess femoral junction", "action": "move"}
  → Wrong:   {"guidance": "Move distally along medial thigh toward mid-thigh", "action": "move"}

RULE 2 — DIRECTION AFTER RP N2→N1 CONFIRMED:
When RP N2→N1 is present in CONFIRMED FINDINGS AND EP N2→N3 is NOT yet confirmed:
  → Q3 is open. The surgeon must move DISTALLY to search for trunk-to-tributary escape.
  → NEVER output "proximally", "toward groin", or any direction back toward the entry point.
  → Always instruct distal movement from the RP confirmation site.

RULE 3 — TYPE 4 CIRCUIT COMPLETE:
When EP N1→N3 is confirmed AND RP N2→N1 is confirmed in CONFIRMED FINDINGS:
  → Type 4 circuit is closed. Output action "complete" IMMEDIATELY.
  → EP N1→N3 means entry is a perforator — the SFJ is competent by definition. Do NOT navigate to SFJ.
  → {"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}

RULE 4 — MANEUVER TAKES ABSOLUTE PRIORITY OVER COMPLETE:
When EP N2→N3 clip + RP N3→N1 clip + RP N2→N1 clip are ALL confirmed AND no elimTest on any EP N2→N3 clip:
  → Output action "maneuver". This is non-negotiable.
  → NEVER output "complete" under these exact conditions. The circuit is ambiguous (Type 3 vs Type 1+2) until
    the elimination test is recorded. "complete" requires elimTest on the EP N2→N3 clip to resolve ambiguity.

━━━ OUTPUT RULES ━━━
First apply CRITICAL PRIORITY RULES above, then determine action type, then write guidance.

action "maneuver" — ONLY when elimination test trigger condition is met (RULE 4).
action "complete" — ONLY when a minimum complete circuit set is confirmed and unambiguous (RULE 3 or elimTest present).
action "move"     — all other cases; one probe-movement instruction ≤12 words.

For action "move": include a position/direction word (proximally / distally / medially / laterally / posteriorly / anteriorly / transversely / deeper / superficially). Name the specific target structure or region. Never mention EP, RP, reflux, Valsalva, shunt type, or any clinical finding.

JSON only — always include both fields:
{"guidance": "<text>", "action": "move"}
{"guidance": "<compression instruction>", "action": "maneuver"}
{"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}

Examples:
{"guidance": "Scan transversely at groin crease to assess femoral junction", "action": "move"}
{"guidance": "Move distally along medial thigh toward mid-thigh", "action": "move"}
{"guidance": "Scan posteriorly at knee level for popliteal junction", "action": "move"}
{"guidance": "Track medial calf distally following saphenous trunk", "action": "move"}
{"guidance": "Rotate probe posteriorly at mid-calf to locate perforator", "action": "move"}
{"guidance": "Follow N3 tributary distally on medial lower calf", "action": "move"}
{"guidance": "Compress tributary at mid-thigh — record whether GSV Doppler changes", "action": "maneuver"}
{"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}"""


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame_at(pos_y_ratio: float) -> Optional[str]:
    """
    Seek to pos_y_ratio × total_frames in the annotated stream video.
    Returns base64 JPEG string, or None on failure.
    """
    from config import STREAM_VIDEO_PATH
    try:
        cap = cv2.VideoCapture(STREAM_VIDEO_PATH)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        idx = min(int(pos_y_ratio * total), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode()
    except Exception as exc:
        logger.error("Frame extraction error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED ACTION DETERMINATION (deterministic, before LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_action(clips: list[dict]) -> tuple[str | None, str | None]:
    """
    Evaluate deterministic circuit-state rules against the confirmed clip set.
    Returns (action, guidance) if a rule fires, (None, None) to fall through to LLM.

    Rules are evaluated in priority order — maneuver before complete.
    """
    def has(flow, ft, tt):
        return any(
            c.get("flow") == flow and c.get("from_type") == ft and c.get("to_type") == tt
            for c in clips
        )

    ep_n1_n2 = has("EP", "N1", "N2")
    ep_n1_n3 = has("EP", "N1", "N3")
    ep_n2_n3 = has("EP", "N2", "N3")
    rp_n2_n1 = has("RP", "N2", "N1")
    rp_n3_n1 = has("RP", "N3", "N1")
    rp_n3_n2 = has("RP", "N3", "N2")

    # Search for the first EP N2→N3 clip that HAS an elimTest value.
    # (Surgeon may add a second EP N2→N3 clip with the elim result, leaving
    # the original clip without one — so we skip empty ones.)
    ep_n2_n3_elim = next(
        (c.get("elimination_test", "") for c in clips
         if c.get("flow") == "EP" and c.get("from_type") == "N2" and c.get("to_type") == "N3"
         and c.get("elimination_test", "")),
        ""
    )
    # True if NO EP N2→N3 clip has an elimTest (maneuver trigger condition)
    ep_n2_n3_no_elim = ep_n2_n3 and not ep_n2_n3_elim

    COMPLETE_MSG = "Circuit mapped — sufficient findings for classification"

    # 1. Maneuver: all three trigger clips present, no elimTest yet on any EP N2→N3
    if ep_n2_n3_no_elim and rp_n3_n1 and rp_n2_n1:
        return "maneuver", "Compress tributary — record whether GSV Doppler changes"

    # 2. Type 3 complete: elimination test recorded as No Reflux
    if ep_n2_n3_elim == "No Reflux":
        return "complete", COMPLETE_MSG

    # 3. Type 1+2 complete: elimination test recorded as Reflux
    if ep_n2_n3_elim == "Reflux":
        return "complete", COMPLETE_MSG

    # 4. Type 4 complete: direct perforator entry + trunk reflux
    if ep_n1_n3 and rp_n2_n1:
        return "complete", COMPLETE_MSG

    # 5. Type 6 complete: perforator-to-perforator, no trunk
    if ep_n1_n3 and rp_n3_n1 and not rp_n2_n1 and not rp_n3_n2:
        return "complete", COMPLETE_MSG

    # 6. Type 1 complete: SFJ/Hunterian entry + trunk reflux, no escape
    if ep_n1_n2 and rp_n2_n1 and not ep_n2_n3:
        return "complete", COMPLETE_MSG

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# STATE MESSAGE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_state_message(
    region: str,
    pos_y: float,
    surface: str,
    leg: str,
    clips: list[dict],
    vlm_summary: str,
) -> str:
    """Format the current probe state as a user message for the LLM."""
    if clips:
        clip_lines = []
        for c in clips:
            flow  = c.get("flow", "?")
            fT    = c.get("from_type", "?")
            tT    = c.get("to_type", "?")
            pY    = c.get("pos_y_ratio", 0.0)
            cleg  = c.get("leg", "?")
            elim  = c.get("elimination_test", "")
            elim_str = f"  elimTest={elim}" if elim else ""
            clip_lines.append(f"  • {flow} {fT}→{tT}  posY={pY:.2f}  {cleg} leg{elim_str}")
        clips_text = "\n".join(clip_lines)
    else:
        clips_text = "  None confirmed yet."

    # Position context hint — tells the model when it is already at a key site
    if 0.04 <= pos_y <= 0.09 and not clips:
        position_hint = (
            "\n>>> POSITION ALERT: Probe is IN the SFJ zone right now (posY "
            f"{pos_y:.2f}). This IS the primary Q1 candidate. "
            "Output a transverse-scan instruction for the groin crease — "
            "do NOT navigate away from this position. <<<"
        )
    else:
        position_hint = ""

    return (
        f"PROBE STATE\n"
        f"Region: {region} | Surface: {surface} | Leg: {leg} | posY: {pos_y:.2f}\n\n"
        f"CONFIRMED FINDINGS\n{clips_text}\n\n"
        f"VLM FRAME ANNOTATION\n{vlm_summary}"
        f"{position_hint}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL WITH HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def call_with_history(
    session: "StreamSession",
    state_message: str,
    api_key: str,
    model: str,
) -> tuple[str, str, str]:
    """
    Call Groq with system prompt + rolling conversation history + current state.
    Updates session.history.
    Returns (guidance_text, raw_response_string, action).
    action is one of: "move", "maneuver", "complete".
    """
    from groq import Groq
    from config import STREAM_HISTORY_WINDOW

    messages = [
        {"role": "system", "content": _STREAM_SYSTEM_PROMPT},
        *session.history,
        {"role": "user", "content": state_message},
    ]

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=80,
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()

    # Parse JSON response
    raw_clean = raw
    if raw_clean.startswith("```"):
        raw_clean = "\n".join(l for l in raw_clean.splitlines() if not l.startswith("```"))
    s, e = raw_clean.find("{"), raw_clean.rfind("}")
    guidance = raw
    action   = "move"
    if s != -1 and e != -1:
        try:
            parsed   = json.loads(raw_clean[s:e + 1])
            guidance = parsed.get("guidance", raw)
            action   = parsed.get("action", "move")
            if action not in ("move", "maneuver", "complete"):
                action = "move"
        except json.JSONDecodeError:
            pass

    session.push_exchange(state_message, raw, window=STREAM_HISTORY_WINDOW)
    return guidance, raw, action


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED VLM + LLM STEP — called from route handler
# ─────────────────────────────────────────────────────────────────────────────

def process_probe_state(
    session: "StreamSession",
    region: str,
    pos_y: float,
    surface: str,
    leg: str,
    force_vlm: bool = False,
    force_llm: bool = False,
) -> dict:
    """
    Given current probe state and session context:
      1. Decide whether to re-run VLM (position changed enough or force_vlm).
      2. Decide whether to re-run LLM (position changed enough or force_llm).
      3. Return a result dict for the WebSocket emit.
    """
    from config import GROQ_API_KEY, GROQ_TEXT_MODEL, GROQ_VISION_MODEL
    from config import STREAM_VLM_THRESHOLD, STREAM_LLM_THRESHOLD
    from vlm_analyzer import analyze_frame

    # ── VLM ──────────────────────────────────────────────────────────────────
    frame_b64: Optional[str] = None
    vlm_dict: Optional[dict] = None

    run_vlm = force_vlm or abs(pos_y - session.last_vlm_pos_y) >= STREAM_VLM_THRESHOLD
    if run_vlm:
        frame_b64 = extract_frame_at(pos_y)
        if frame_b64:
            assessment = analyze_frame(frame_b64, region, leg, "image/jpeg")
            session.last_vlm_summary = assessment.summary()
            session.last_vlm_pos_y   = pos_y
            vlm_dict = assessment.to_dict()

    vlm_summary = session.last_vlm_summary

    # ── LLM ──────────────────────────────────────────────────────────────────
    guidance  = None
    raw       = None
    action    = None
    state_msg = None

    run_llm = force_llm or abs(pos_y - session.last_llm_pos_y) >= STREAM_LLM_THRESHOLD
    if run_llm:
        state_msg = build_state_message(region, pos_y, surface, leg, session.clips, vlm_summary)

        # --- Rule-based pre-check: deterministic circuit-state decisions ---
        rule_action, rule_guidance = _rule_based_action(session.clips)

        if rule_action:
            # Skip LLM entirely for maneuver/complete — rules are authoritative
            guidance = rule_guidance
            raw      = f"[rule-based] action={rule_action}"
            action   = rule_action
            session.last_llm_pos_y = pos_y
            if action in ("complete", "maneuver"):
                session.last_llm_pos_y = -1.0
            session.push_thinking(pos_y, region, state_msg, raw, guidance)
        else:
            try:
                guidance, raw, action = call_with_history(session, state_msg, GROQ_API_KEY, GROQ_TEXT_MODEL)
                session.last_llm_pos_y = pos_y
                # Reset position cache after terminal actions so the next probe_move
                # always triggers a fresh LLM call (prevents timeout on final confirmation step)
                if action in ("complete", "maneuver"):
                    session.last_llm_pos_y = -1.0
                session.push_thinking(pos_y, region, state_msg, raw, guidance)
            except Exception as exc:
                logger.error("LLM error: %s", exc)
                guidance = "Unable to generate guidance."
                raw      = str(exc)
                action   = "move"

    return {
        "guidance":    guidance,
        "action":      action,
        "vlm":         vlm_dict,
        "vlm_summary": vlm_summary,
        "region":      region,
        "pos_y":       pos_y,
        "thinking": {
            "state": state_msg,
            "raw":   raw,
        } if state_msg else None,
    }
