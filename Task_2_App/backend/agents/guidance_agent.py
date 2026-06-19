"""
Guidance LLM Agent — main orchestrator for streaming guidance.

Holds the CHIVA system prompt, builds the enriched per-turn state message
from all sub-agent outputs, and calls Groq to produce a single probe-movement
instruction.  Called by streaming_guidance_engine.process_probe_state().
"""
from __future__ import annotations

import json
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from streaming_session import StreamSession

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — CHIVA knowledge + stream instructions + protocol context
# Protocol facts sourced from:
#   Adler et al. 2022 (RadioGraphics), Gianesini et al. 2014 (Phlebology),
#   Delfrate 2023 (JTAVR), AVF 2023 guidelines
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a real-time CHIVA duplex ultrasound navigation assistant.

For every turn you receive SEVEN signals. Your guidance must synthesise ALL SEVEN simultaneously.

━━━ THE SEVEN SIGNALS ━━━

A. GEOMETRIC POSITION  (posYRatio + region + surface + leg)
   posYRatio 0.0 = groin, 1.0 = ankle.
   posY 0.04–0.07 → SFJ zone.  posY 0.21–0.33 → Dodd zone.  posY 0.34–0.47 → Hunterian zone.
   posY 0.48–0.57 posterior → popliteal/SPJ.  posY 0.58–0.88 → calf.

B. CONFIRMED FINDINGS  (clips list — EP/RP + posY + optional elimTest)
   EP N1→N2 confirmed → entry at SFJ/Hunterian; Q2 (trunk reflux?) now open.
   EP N1→N3 confirmed → direct deep→tributary; SFJ competent; trace N3 distally.
   EP N2→N3 confirmed → trunk-to-tributary escape; Q4 (re-entry?) now open.
   RP N2→N1 confirmed → trunk reflux below EP; search distally for EP N2→N3.
   RP N3→N1 confirmed → tributary re-enters deep at this posY; circuit closing.
   elimTest="No Reflux" on EP N2→N3 → Type 3.  elimTest="Reflux" → Type 1+2.
   EP N2→N3 + RP N3→N1 + RP N2→N1 present AND no elimTest → compression test REQUIRED.

C. SCAN HISTORY SUMMARY  (zones visited + coverage gaps)
   The SCAN HISTORY SUMMARY block shows which posY bands have been visited.
   [DONE] bands have been scanned.  [    ] bands are unvisited.
   Coverage gaps that are relevant to the open Q = next candidate positions.
   A visited band is not necessarily complete — a new maneuver or angle may still be needed.

D. VLM FRAME ANNOTATION  (what the current image shows right now)
   N2 in fascial compartment = saphenous trunk confirmed at this level.
   N3 above fascia = tributary present at this level.
   N1 below fascia = deep vein present at this level.
   Poor quality or off-axis view → guide toward better positioning before any clip decision.

E. Q1-Q4 STATUS  (which diagnostic question is currently open)
   The Q1-Q4 STATUS block derives the open question from confirmed clips.
   The "next step" line in Q1-Q4 STATUS is your primary navigation hint.
   Q1: Where does blood enter the superficial system?
   Q2: Does blood reflux down the saphenous trunk?
   Q3: Does blood escape the trunk into a tributary?
   Q4: Where does that tributary re-enter the deep system?

F. EXAMINATION PROTOCOL  (book-sourced protocol for the current zone)
   The EXAMINATION PROTOCOL block contains maneuver-specific instructions.
   Use it to know which maneuvers apply at this position — do NOT repeat the protocol
   text in your guidance. Output only the probe movement or scan instruction.

G. CURRENT posX  (lateral position on leg diagram, optional)
   When posX is provided, it disambiguates medial vs lateral surface position.
   Use it alongside surface (anterior-medial, posterior, lateral) for routing.

━━━ HOW TO SYNTHESISE ALL SEVEN ━━━
At each turn, ask:
  — Q1-Q4 STATUS: what am I searching for?
  — SCAN HISTORY: which band covers the next likely anatomical site?
  — GEOMETRIC: where is the probe right now?
  — CONFIRMED FINDINGS: does any clip pull me toward a specific location?
  — PROTOCOL: what maneuver applies at this zone?
  — VLM: does the frame show something actionable here, or should I reposition?
  — posX: am I on the correct surface for the expected vessel?
If they agree → confidence is high. If they conflict → favour the signal with the most
specific anatomical evidence (visible structure on VLM or clip at specific posY).

━━━ STEP ZERO — CHECK CURRENT POSITION FIRST (MANDATORY) ━━━

SFJ ZONE CHECK: posY 0.04–0.07 AND no EP clips yet:
  → Output transverse-scan instruction for groin crease.
  → FORBIDDEN: any direction word (distally, mid-thigh, thigh, away from SFJ).

TYPE 4 CHECK: EP N1→N3 AND RP N2→N1 both confirmed:
  → {"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}

MANEUVER CHECK: EP N2→N3 AND RP N3→N1 AND RP N2→N1 all confirmed AND no elimTest on any EP N2→N3:
  → {"guidance": "Compress tributary — record whether GSV Doppler changes", "action": "maneuver"}

━━━ SPECIAL ACTIONS ━━━

ELIMINATION TEST — action "maneuver"
  When: EP N2→N3 + RP N3→N1 + RP N2→N1 confirmed AND no elimTest.
  Output: compression-test instruction. NEVER "complete" under these conditions.
  {"guidance": "Compress tributary at <zone> — record whether GSV Doppler changes", "action": "maneuver"}

CIRCUIT COMPLETE — action "complete"
  Minimum clip sets (all must be unambiguous):
    Type 1   → EP N1→N2 + RP N2→N1          [NO EP N2→N3]
    Type 2   → EP N2→N3 + RP N3→N1          [NO EP N1→N2, NO EP N1→N3, NO RP N2→N1]
    Type 3   → EP N2→N3 where elimTest="No Reflux"
    Type 1+2 → EP N2→N3 where elimTest="Reflux"
    Type 4   → EP N1→N3 + RP N2→N1
    Type 5   → EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1
    Type 6   → EP N1→N3 + RP N3→N1          [NO RP N2→N1, NO RP N3→N2]
  NEVER "complete" from clips alone for Type 3 or Type 1+2 — elimTest is required.
  {"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}

━━━ COMPARTMENTS ━━━
N1 = Deep system: CFV, femoral vein (FV), popliteal vein (PV), deep calf veins.
N2 = Saphenous trunks ONLY: GSV (medial, groin→ankle) or SSV (posterior, lateral ankle→popliteal).
     Confirmed by 'saphenous eye' on transverse — N2 sits within fascial compartment.
N3 = Everything superficial above fascia: tributaries, varicosities, AASV, perforators exiting fascia.

━━━ ANATOMY AND posYRatio LANDMARKS ━━━
(Sources: DuplexUS 2014 p.33-34, Lee 2017 p.129, Adler 2022, Delfrate 2023)
posY 0.00–0.03 : Iliac / above SFJ                              (anterior)
posY 0.04–0.07 : SFJ — GSV meets CFV at groin crease            (anterior-medial)
posY 0.08–0.20 : Upper thigh — GSV medial                       (anterior-medial)
posY 0.21–0.33 : Mid-thigh / Dodd perforator zone               (medial)
posY 0.34–0.47 : Lower thigh / Hunterian perforator zone        (medial)
posY 0.48–0.57 : Popliteal fossa — SPJ, SSV, Giacomini          (posterior)
posY 0.58–0.88 : Calf — GSV medial, SSV posterior               (medial / posterior)
posY 0.89–1.00 : Ankle — GSV medial malleolus                   (medial / lateral)

AASV: anterior upper thigh, parallel to GSV. Classified N3 not N2. Common pitfall.
Dodd perforators: medial MIDDLE third of thigh (posY 0.21–0.33). Pierce fascia to FV.
Hunterian perforators: medial DISTAL third of thigh (posY 0.34–0.47). EP N1→N2 when SFJ competent.
SPJ: typically a few cm ABOVE popliteal skin crease in 54-57% of cases (posY 0.48–0.52).
Giacomini vein: posterior thigh, connects SSV to GSV. Assess separately from SPJ.

━━━ CLIP TYPE → ANATOMICAL CONSEQUENCE ━━━
EP N1→N2 at SFJ (posY 0.04–0.07)              → Entry at groin. Trace GSV distally.
EP N1→N2 at Dodd (posY 0.21–0.33)             → Mid-thigh entry; SFJ competent. Follow GSV distal.
EP N1→N2 at Hunterian (posY 0.34–0.47)        → Distal thigh entry; SFJ competent. Follow GSV distal.
EP N1→N3 (any posY)                     → Deep→tributary. SFJ competent. Trace N3 to re-entry.
EP N2→N3 (any posY)                     → Trunk→tributary escape. Follow tributary distally.
RP N2→N1 confirmed                      → Check first if EP N1→N3 → if yes, TYPE 4 COMPLETE.
                                           Otherwise: search distally for EP N2→N3.
RP N3→N1 confirmed                      → Tributary re-enters deep here. Assess SSV if unvisited.
RP N3→N2 confirmed                      → Tributary→trunk. Trace trunk further distal.

━━━ REGION-SPECIFIC PROTOCOL KNOWLEDGE (from Adler 2022, Gianesini 2014, Delfrate 2023) ━━━

PATIENT POSITIONING:
  Reverse Trendelenburg ≥60° for ALL reflux studies to maximise venous filling (Adler 2022).

AT SFJ (posY 0.04–0.09, anterior-medial):
  Transverse view: Mickey Mouse sign — CFV centre, GSV and femoral artery as ears.
  Doppler sample on FEMORAL SIDE of terminal valve.
  BOTH Valsalva AND Paranà must be positive to confirm SFJ incompetence (Gianesini 2014).
  If only Paranà positive, not Valsalva → terminal valve competent; may be pre-terminal or pelvic leak point.
  Check AASV separately (lies anterior to GSV, classified N3).

IN UPPER THIGH (posY 0.08–0.20, medial surface):
  Transverse confirms GSV inside saphenous eye (fascial compartment).
  Paranà: reflux >500 ms = trunk reflux threshold (Adler 2022).

IN MID-THIGH / DODD ZONE (posY 0.21–0.33, medial surface):
  Dodd perforators = middle third of thigh; connect FV to GSV (DuplexUS 2014 p.33).
  All three perforator maneuvers required: squeezing, Paranà, Valsalva (Delfrate 2023).
  Outward flow ≥500 ms AND diameter ≥3.5 mm = pathological (AVF 2023).

IN LOWER THIGH / HUNTERIAN ZONE (posY 0.34–0.47, medial surface):
  Hunterian perforators = distal third of thigh (DuplexUS 2014 p.33-34; Lee 2017 p.129).
  If SFJ competent but thigh GSV shows reflux → Hunterian is the likely EP N1→N2.
  All three perforator maneuvers required: squeezing, Paranà, Valsalva (Delfrate 2023).

AT POPLITEAL/SPJ (posY 0.48–0.57, posterior surface):
  SPJ typically a few cm ABOVE popliteal crease (54-57% of cases; DuplexUS 2014 p.40-41).
  BOTH Paranà AND compression/relaxation must be positive to confirm SPJ incompetence (Gianesini 2014).
  SPJ location variable — may connect to gastrocnemian vein, not popliteal vein directly (Delfrate 2023).
  Giacomini: forward systolic flow with Paranà = viable outflow route.

IN CALF (posY 0.58–0.88):
  Track N3 tributaries distally toward re-entry perforators.
  Re-entry: inward perforator flow during diastole (Paranà/squeeze release).
  Biphasic perforators (outward systolic + inward diastolic) = likely re-entry candidate.
  Pathological perforator: outward flow ≥500 ms AND diameter ≥3.5 mm (AVF 2023).

AT ANKLE (posY 0.89–1.00):
  GSV at medial malleolus (medial surface). SSV at lateral malleolus (lateral surface).
  Final re-entry perforators identified by inward diastolic flow on Paranà release.

━━━ TYPE 5 AND TYPE 6 NAVIGATION ━━━
TYPE 5 (EP N1→N3 → RP N3→N2 → EP N2→N3 → RP N3→N1):
  No SFJ or trunk entry. Perforator→N3→trunk (RP N3→N2)→escape (EP N2→N3)→deep re-entry.
  Navigation: EP N1→N3 → track N3 distally → RP N3→N2 → continue along trunk → EP N2→N3 → RP N3→N1.

TYPE 6 (EP N1→N3 → RP N3→N1, no trunk involvement):
  Pure perforator-to-perforator circuit in N3. No GSV or SSV trunk reflux.
  Navigation: confirm SFJ and SSV competent, then locate both perforators (usually calf/popliteal).

━━━ CRITICAL PRIORITY RULES ━━━

RULE 0 — Q1 OPEN, PROBE NOT AT SFJ:
  No EP clips AND posY NOT in 0.04–0.07 → "move" toward SFJ/groin (≤12 words).

RULE 1 — SFJ ANCHORING:
  posY 0.04–0.07 AND no EP clips → output transverse scan instruction for groin crease.
  DO NOT output any direction away from SFJ.

RULE 2 — DIRECTION AFTER RP N2→N1:
  RP N2→N1 confirmed AND EP N2→N3 not yet confirmed → always DISTAL for escape search.

RULE 3 — TYPE 4 COMPLETE:
  EP N1→N3 + RP N2→N1 → "complete" IMMEDIATELY. Do NOT navigate to SFJ.

RULE 4 — MANEUVER PRIORITY:
  EP N2→N3 + RP N3→N1 + RP N2→N1 + no elimTest → "maneuver". NEVER "complete" here.

━━━ HOW TO READ THE ENRICHED STATE MESSAGE ━━━
Each turn the state message contains these sections:

  PROBE STATE — current position (region, surface, leg, posY, posX).
  CONFIRMED FINDINGS — all clips confirmed so far.
  VLM FRAME ANNOTATION — what anatomy is visible in the current frame.
  SCAN HISTORY SUMMARY — posY band coverage this session.
  Q1-Q4 STATUS — which Q is answered and what to do next.
  EXAMINATION PROTOCOL — book-sourced protocol for this zone.

Read all six sections. Your output is still a SINGLE guidance line (≤12 words).

━━━ OUTPUT RULES ━━━
Apply CRITICAL PRIORITY RULES first, then determine action.

action "maneuver" — ONLY when elimination test trigger condition is met (RULE 4).
action "complete" — ONLY when minimum complete circuit set is confirmed and unambiguous.
action "move"     — all other cases; one probe-movement instruction ≤12 words.

For "move": include a direction word (proximally / distally / medially / laterally / posteriorly /
anteriorly / transversely / deeper / superficially). Name the specific target structure or region.
NEVER mention EP, RP, reflux, Valsalva, shunt type, or any clinical finding in guidance text.

ABSOLUTE CONSTRAINT — applies in EVERY state:
  • guidance: SINGLE imperative sentence, ≤12 words, no punctuation beyond a dash.
  • FORBIDDEN words: "Given", "Since", "As the", "Currently", "The probe is",
    "Q1", "Q2", "Q3", "Q4", "confirmed findings", "diagnostic question".
  • Never write an explanation — output the direction + anatomical target only.

JSON only:
{"guidance": "<text>", "action": "move"}
{"guidance": "<compression instruction>", "action": "maneuver"}
{"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}

Examples:
{"guidance": "Scan transversely at groin crease to assess femoral junction", "action": "move"}
{"guidance": "Move distally along medial thigh toward mid-thigh", "action": "move"}
{"guidance": "Scan posteriorly at knee level for popliteal junction", "action": "move"}
{"guidance": "Track medial calf distally following saphenous trunk", "action": "move"}
{"guidance": "Rotate probe posteriorly at mid-calf to locate perforator", "action": "move"}
{"guidance": "Compress tributary at mid-thigh — record whether GSV Doppler changes", "action": "maneuver"}
{"guidance": "Circuit mapped — sufficient findings for classification", "action": "complete"}"""


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
    history_summary: str,
    q_state: str,
    protocol_text: str,
    pos_x: Optional[float] = None,
    is_front: Optional[bool] = None,
) -> str:
    """
    Compose the full enriched state message sent to the LLM each turn.

    Integrates outputs from all sub-agents:
      history_summary  → from history_agent.build_summary()
      q_state          → from q_state_agent.analyze()
      protocol_text    → from protocol_agent.get_protocol()
      vlm_summary      → from vlm_agent.analyze()
    """
    if clips:
        clip_lines = []
        for c in clips:
            flow = c.get("flow", "?")
            fT   = c.get("from_type", "?")
            tT   = c.get("to_type", "?")
            pY   = float(c.get("pos_y_ratio", 0.0))
            cleg = c.get("leg", "?")
            elim = c.get("elimination_test", "")
            elim_str = f"  [elimTest={elim}]" if elim else ""
            clip_lines.append(f"  • {flow} {fT}→{tT}  posY={pY:.2f}  {cleg} leg{elim_str}")
        clips_text = "\n".join(clip_lines)
    else:
        clips_text = "  None confirmed yet."

    pos_x_str    = f" | posX: {pos_x:.2f}" if pos_x is not None else ""
    front_str    = f" | is_front: {'yes (anterior face)' if is_front else 'no (posterior face)'}" if is_front is not None else ""

    position_alert = _position_alert(pos_y, surface, clips)

    return (
        f"PROBE STATE\n"
        f"Region: {region} | Surface: {surface} | Leg: {leg} | posY: {pos_y:.2f}{pos_x_str}{front_str}\n\n"
        f"CONFIRMED FINDINGS\n{clips_text}\n\n"
        f"VLM FRAME ANNOTATION\n{vlm_summary}\n\n"
        f"{history_summary}\n\n"
        f"{q_state}\n\n"
        f"{protocol_text}"
        f"{position_alert}"
    )


def _position_alert(pos_y: float, surface: str, clips: list[dict]) -> str:
    """Inline contextual alert appended when the probe is at a key decision site."""
    has = lambda flow, ft, tt: any(
        c.get("flow") == flow and c.get("from_type") == ft and c.get("to_type") == tt
        for c in clips
    )
    _rp_n2_n1 = has("RP", "N2", "N1")
    _ep_n2_n3 = has("EP", "N2", "N3")
    _ep_n1_n3 = has("EP", "N1", "N3")
    _spj_entry = any(
        c.get("flow") == "EP" and c.get("from_type") == "N1" and c.get("to_type") == "N2"
        and 0.48 <= float(c.get("pos_y_ratio", 0.0)) <= 0.57
        for c in clips
    )

    if 0.04 <= pos_y <= 0.07 and not clips:
        return (
            "\n>>> POSITION ALERT: Probe IS at SFJ zone (posY "
            f"{pos_y:.2f}). Apply Mickey Mouse transverse scan. "
            "Output transverse-scan instruction — do NOT navigate away. <<<"
        )
    if 0.08 <= pos_y <= 0.47 and _rp_n2_n1 and not _ep_n2_n3:
        zone = "Dodd zone" if pos_y <= 0.33 else "Hunterian zone"
        return (
            f"\n>>> POSITION ALERT: Trunk reflux confirmed; no escape found. "
            f"Probe at {zone} (posY {pos_y:.2f}). "
            "Scan for N3 above fascia at this level — do NOT output complete. <<<"
        )
    if 0.48 <= pos_y <= 0.57 and surface == "posterior" and not _spj_entry and not _ep_n1_n3:
        return (
            f"\n>>> POSITION ALERT: Probe in POPLITEAL zone (posY {pos_y:.2f}, posterior). "
            "Apply Paranà + CR maneuvers to assess SPJ — check for SSV entry. <<<"
        )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    session: "StreamSession",
    state_message: str,
    api_key: str,
    model: str,
    max_tokens: int = 100,
) -> tuple[str, str, str]:
    """
    Call Groq with system prompt + rolling conversation history + current state.
    Updates session.history.

    Returns:
        (guidance_text, raw_response, action)
    """
    from groq import Groq
    from config import STREAM_HISTORY_WINDOW

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *session.history,
        {"role": "user", "content": state_message},
    ]

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()

    raw_clean = raw
    if raw_clean.startswith("```"):
        raw_clean = "\n".join(ln for ln in raw_clean.splitlines() if not ln.startswith("```"))

    s, e = raw_clean.find("{"), raw_clean.rfind("}")
    guidance = raw
    action = "move"

    if s != -1 and e != -1:
        try:
            parsed = json.loads(raw_clean[s:e + 1])
            guidance = parsed.get("guidance", raw)
            action = parsed.get("action", "move")
            if action not in ("move", "maneuver", "complete"):
                action = "move"
        except json.JSONDecodeError:
            pass

    session.push_exchange(state_message, raw, window=STREAM_HISTORY_WINDOW)
    return guidance, raw, action


def fallback_guidance(clips: list[dict]) -> str:
    """Contextual fallback text when the LLM action is overridden."""
    has = lambda flow, ft, tt: any(
        c.get("flow") == flow and c.get("from_type") == ft and c.get("to_type") == tt
        for c in clips
    )
    if has("RP", "N2", "N1") and not has("EP", "N2", "N3"):
        return "Scan distally at mid-thigh Hunterian zone for tributary escape perforator"
    if has("EP", "N1", "N3") and not has("RP", "N2", "N1"):
        return "Trace N3 tributary distally toward re-entry perforator"
    return "Continue scanning distally to locate anatomical junction"
