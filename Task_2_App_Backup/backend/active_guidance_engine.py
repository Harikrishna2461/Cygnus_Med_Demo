"""
Active guidance engine — Groq LLM only.

System prompt: full CHIVA clinical logic sourced verbatim from Task-1 nl_interpreter.py:
  - _NL_TO_CHIVA_PROMPT  : compartments, clip types, critical rules 1/2/2C/3
  - _CONVERSATIONAL_PROMPT: shunt hemodynamics, junctions, scanning manoeuvres
  - _SUFFICIENCY_PROMPT  : Q1-Q4 framework

_chiva_component_status(): Python-side Q1-Q4 analysis + circuit-based location prediction.
  Given confirmed clips, predicts anatomical location of the NEXT expected EP/RP and
  which region/probe position to scan, based on CHIVA circuit logic.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from probe_localizer import ProbeLocation, WrongRegionResult, check_wrong_region
from vlm_analyzer import UltrasoundAssessment

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PER-SESSION CONVERSATION HISTORY
# Stores (user_prompt, assistant_raw) pairs so each LLM call sees prior steps.
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_HISTORY_WINDOW = 6   # pairs of (user, assistant) kept per session
_history_store: dict[str, list[dict]] = {}
_history_lock = threading.Lock()


def _get_history(session_id: str) -> list[dict]:
    with _history_lock:
        return _history_store.setdefault(session_id, [])


def _push_history(session_id: str, user_prompt: str, assistant_raw: str) -> None:
    with _history_lock:
        hist = _history_store.setdefault(session_id, [])
        hist.append({"role": "user",      "content": user_prompt})
        hist.append({"role": "assistant", "content": assistant_raw})
        cap = _ACTIVE_HISTORY_WINDOW * 2
        if len(hist) > cap:
            _history_store[session_id] = hist[-cap:]


def clear_history(session_id: str) -> None:
    with _history_lock:
        _history_store.pop(session_id, None)

# ─────────────────────────────────────────────────────────────────────────────
# CHIVA SYSTEM PROMPT
# Every fact below is sourced from Task-1 nl_interpreter.py.
# Section references are given inline.
# ─────────────────────────────────────────────────────────────────────────────

_CHIVA_SYSTEM_PROMPT = """You are a real-time CHIVA duplex ultrasound guidance assistant.
You receive: (1) current probe position, (2) Q1-Q4 diagnostic status + circuit prediction
derived from confirmed EP/RP clips, (3) VLM frame annotation context.
Write ONE line (≤12 words) — the single next action for the sonographer.

━━━ COMPARTMENTS (source: nl_interpreter.py _NL_TO_CHIVA_PROMPT lines 109-117) ━━━
N1 = Deep venous system ONLY: femoral vein, popliteal vein, deep calf veins.
N2 = Saphenous trunk ONLY: GSV (groin→medial ankle) or SSV (lateral ankle→popliteal fossa).
     N2 applies only when the named saphenous trunk is the vessel. Sits within fascial compartment.
N3 = Everything else superficial: tributaries, branches, varicosities, AASV, perforators.
     "Superficial veins" without naming GSV/SSV = N3, not N2.

━━━ FLOW NOTATION (source: nl_interpreter.py _NL_TO_CHIVA_PROMPT lines 135-137) ━━━
EP = Antegrade — forward flow; blood entering superficial system or moving trunk→tributary.
RP = Retrograde — backward reflux; pathological, away from heart, valve failure.

━━━ ANATOMY & POSITIONS (source: nl_interpreter.py lines 119-133, 139-147) ━━━
GSV runs medially from SFJ (groin) to medial malleolus within the saphenous fascial compartment.
SSV runs posteriorly from lateral malleolus to SPJ (popliteal fossa).
SFJ — GSV joins common femoral vein at groin crease. posYRatio ≈ 0.04–0.09.
SPJ — SSV joins popliteal vein behind knee. posYRatio ≈ 0.40–0.50.
Hunterian perforators — medial thigh, posYRatio 0.10–0.35. When incompetent: EP N1→N2.
Boyd/paratibial perforators — upper medial calf.
Posterior tibial perforators — medial calf/ankle.
AASV (Anterior Accessory Saphenous Vein) — anterior/parallel to GSV in upper thigh. Classified N3
     unless explicitly called a saphenous trunk. Common duplex pitfall — may be misidentified as GSV.

posYRatio scale (0 = groin, 1 = ankle):
  SFJ/groin:           0.04–0.09  → SFJ region
  Upper thigh:         0.10–0.20  → GSV-THI proximal
  Mid-thigh/Hunterian: 0.21–0.35  → GSV-THI mid
  Knee/popliteal:      0.40–0.55  → GSV-THI distal / SPJ
  Calf:                0.60–0.80  → GSV-CAL / SSV
  Ankle:               0.85–1.00  → GSV-CAL distal

━━━ ALL CLIP TYPES (source: nl_interpreter.py lines 330-393) ━━━
EP N1→N2 — Deep system → saphenous trunk. SFJ/SPJ/Hunterian INCOMPETENT.
            Named junction has failed. Blood crosses from N1 into N2.
            Covers: SFJ (posY≈0.06), SPJ (posY≈0.45), Hunterian (posY≈0.25).
EP N2→N2 — Perforating vessel → saphenous trunk. SFJ COMPETENT.
            Not a named junction — a perforator feeding mid-GSV. SFJ/SPJ valves still work.
EP N2→N3 — Saphenous trunk → tributary (escape/overflow point).
            Blood leaves GSV/SSV and enters any side vessel or branch.
EP N1→N3 — Deep system → tributary DIRECTLY, bypassing saphenous trunk.
            SFJ is competent. Source: pelvic/pudendal/gluteal/perforating vein.
RP N2→N1 — Saphenous trunk reflux backward (toward foot). GSV/SSV valve failure.
            Separate from EP N1→N2 — both can coexist (Type 1). Do not assume one implies the other.
RP N3→N2 — Tributary carries blood backward toward the saphenous trunk.
RP N3→N1 — Tributary carries blood backward into deep system via perforator.

━━━ CRITICAL DISTINCTION: EP N2→N2 vs EP N1→N2 (source: nl_interpreter.py lines 150-195) ━━━
EP N1→N2 = deep SYSTEM delivers blood into GSV — SFJ/Hunterian junction FAILED.
EP N2→N2 = perforator inserts into GSV — SFJ competent; "deep" describes the vessel's anatomy, not the blood source.
The word "deep" alone (e.g. "deep perforating vessel") does NOT make it EP N1→N2.
Only use EP N1→N2 when: "SFJ incompetent", "femoral vein feeds GSV", "Hunterian perforator incompetent",
"deep venous blood enters GSV", OR "[perforator] connects the DEEP SYSTEM to the GSV".

━━━ CRITICAL RULE: EP N1→N2 ≠ RP N2→N1 (source: nl_interpreter.py lines 218-221) ━━━
SFJ incompetence alone does NOT imply GSV trunk reflux.
EP N1→N2 and RP N2→N1 are separate findings — both must be individually confirmed.

━━━ TYPE 3 CONDUIT RULE (source: nl_interpreter.py lines 236-264) ━━━
When EP N1→N2 + EP N2→N3 are both confirmed AND no reflux below the escape point is seen:
Do NOT assume RP N2→N1. The GSV may be acting as a CONDUIT from SFJ to the escape point only.
Type 3 = EP N1→N2 + EP N2→N3 + RP N3 — NO RP N2→N1.
Only record RP N2→N1 if retrograde trunk flow is confirmed BELOW the escape point (posY > escape posY).

━━━ TYPE 1 vs TYPE 4 (source: nl_interpreter.py lines 286-323) ━━━
Both produce RP N2→N1 (trunk reflux). Distinguished only by the ENTRY clip:
  Type 1 entry = EP N1→N2: deep blood enters the SAPHENOUS TRUNK at SFJ/Hunterian/SPJ.
  Type 4 entry = EP N1→N3: deep blood enters a TRIBUTARY directly — SFJ is competent.

━━━ SHUNT CIRCUITS (source: nl_interpreter.py _CONVERSATIONAL_PROMPT lines 725-745) ━━━
Type 1   = EP N1→N2 + RP N2→N1
Type 2A  = EP N2→N3 only
Type 2B  = EP N2→N2 + RP N3 (no RP N2→N1)
Type 2C  = EP N2→N2 + RP N3 + RP N2→N1
Type 3   = EP N1→N2 + EP N2→N3 + RP N3 (no RP N2→N1 — GSV conduit)
Type 1+2 = EP N1→N2 + EP N2→N3 + RP N3 + RP N2→N1
Type 4   = EP N1→N3 + RP N2→N1
Type 5   = EP N1→N3 + RP N3

━━━ Q1-Q4 FRAMEWORK (source: nl_interpreter.py _SUFFICIENCY_PROMPT lines 486-520) ━━━
Q1: Where does blood enter the superficial system?
    → SFJ/SPJ/Hunterian incompetent (EP N1→N2), or perforator entry (EP N2→N2),
      or GSV overflow (EP N2→N3), or direct deep-to-tributary (EP N1→N3).
Q2: Does blood travel backward through the GSV trunk? (required when Q1=SFJ/Hunterian/perforator)
    → YES = RP N2→N1 confirmed. NO = explicitly ruled out.
Q3: Does blood escape from GSV into a tributary? (required when Q1=SFJ/Hunterian/perforator)
    → YES = EP N2→N3 confirmed. NO = explicitly ruled out.
Q4: Does blood travel backward through that tributary? (required when Q3=YES)
    → YES = RP N3→N2 or RP N3→N1 confirmed.

Do not add findings not present in the data.
Output ONLY a probe movement or scan location instruction — where to move or what to look at next.
Never mention techniques (Valsalva, calf augmentation), clinical findings, or explanations.
Respond only with valid JSON: {"guidance": "<≤12 words>"}"""


# ── posYRatio → readable anatomical level ─────────────────────────────────────

def _posY_to_level(pos: float) -> str:
    if pos <= 0.09:  return f"SFJ/groin (posY={pos:.2f})"
    if pos <= 0.20:  return f"upper thigh (posY={pos:.2f})"
    if pos <= 0.35:  return f"mid-thigh/Hunterian (posY={pos:.2f})"
    if pos <= 0.55:  return f"knee/popliteal (posY={pos:.2f})"
    if pos <= 0.80:  return f"calf (posY={pos:.2f})"
    return f"ankle (posY={pos:.2f})"


_CLIP_MEANING: dict[tuple[str, str, str], str] = {
    ("EP", "N1", "N2"): "deep system → saphenous trunk (SFJ/Hunterian INCOMPETENT)",
    ("EP", "N2", "N2"): "perforator → saphenous trunk (SFJ competent)",
    ("EP", "N2", "N3"): "saphenous trunk → tributary (escape point)",
    ("EP", "N1", "N3"): "deep system → tributary directly (SFJ competent)",
    ("RP", "N2", "N1"): "saphenous trunk reflux backward (trunk toward foot)",
    ("RP", "N3", "N2"): "tributary reflux toward saphenous trunk",
    ("RP", "N3", "N1"): "tributary reflux re-enters deep system",
}


def _ep_rp_summary(clips: list[dict]) -> str:
    if not clips:
        return "None confirmed yet."
    lines = []
    for c in clips:
        flow = c.get("flow", "?")
        fT   = c.get("fromType", "?")
        tT   = c.get("toType", "?")
        pos  = c.get("posYRatio")
        leg  = c.get("legSide", "?")
        level   = _posY_to_level(pos) if pos is not None else "unknown level"
        meaning = _CLIP_MEANING.get((flow, fT, tT), "blood flow event")
        lines.append(f"  {flow} {fT}→{tT}  at {level}  {leg} leg  — {meaning}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Q1-Q4 STATUS + CIRCUIT-BASED LOCATION PREDICTION
# Python-side analysis of confirmed clips → tells LLM exactly which Q is
# missing and WHERE on the anatomy to find it, based on CHIVA circuit logic.
# Source: Task-1 _SUFFICIENCY_PROMPT Q1-Q4 framework + _NL_TO_CHIVA_PROMPT
#         critical rules + _CONVERSATIONAL_PROMPT shunt hemodynamics.
# ─────────────────────────────────────────────────────────────────────────────

def _chiva_component_status(clips: list[dict]) -> str:
    if not clips:
        return (
            "No clips confirmed yet.\n"
            "Q1 (entry point): NOT CONFIRMED — move probe to groin crease, anterior thigh, posY≈0.06. "
            "Look for N2 in fascial compartment. If N1 (deep vein) visible alongside N2 at that level, confirm SFJ junction."
        )

    def has(flow: str, fT: str, tT: str) -> bool:
        return any(c.get("flow") == flow and c.get("fromType") == fT and c.get("toType") == tT for c in clips)

    def first(flow: str, fT: str, tT: str) -> Optional[dict]:
        return next((c for c in clips if c.get("flow") == flow and c.get("fromType") == fT and c.get("toType") == tT), None)

    def first_rp_n3() -> Optional[dict]:
        return next((c for c in clips if c.get("flow") == "RP" and c.get("fromType") == "N3"), None)

    has_ep_n1_n2 = has("EP", "N1", "N2")
    has_ep_n2_n2 = has("EP", "N2", "N2")
    has_ep_n2_n3 = has("EP", "N2", "N3")
    has_ep_n1_n3 = has("EP", "N1", "N3")
    has_rp_n2_n1 = has("RP", "N2", "N1")
    has_rp_n3_n2 = has("RP", "N3", "N2")
    has_rp_n3_n1 = has("RP", "N3", "N1")
    has_rp_n3    = has_rp_n3_n2 or has_rp_n3_n1

    lines: list[str] = []

    # ── Q1: Entry point ───────────────────────────────────────────────────────
    if has_ep_n1_n2:
        c = first("EP", "N1", "N2")
        lines.append(
            f"Q1 CONFIRMED — EP N1→N2: SFJ/Hunterian incompetent at {_posY_to_level(c.get('posYRatio', 0.06))} "
            f"({c.get('legSide','?')} leg). Deep blood entering GSV (N2 in fascial compartment)."
        )
    elif has_ep_n2_n2:
        c = first("EP", "N2", "N2")
        lines.append(
            f"Q1 CONFIRMED — EP N2→N2: Perforator feeds GSV at {_posY_to_level(c.get('posYRatio', 0.25))} "
            f"({c.get('legSide','?')} leg). SFJ is competent."
        )
    elif has_ep_n1_n3:
        c = first("EP", "N1", "N3")
        lines.append(
            f"Q1 CONFIRMED — EP N1→N3: Deep system enters tributary directly at "
            f"{_posY_to_level(c.get('posYRatio', 0.25))} ({c.get('legSide','?')} leg). "
            "SFJ competent. Pattern consistent with Type 4 or 5."
        )
    elif has_ep_n2_n3:
        lines.append(
            "Q1 PARTIAL — EP N2→N3 (tributary escape) confirmed. SFJ entry not yet assessed. "
            "Move probe to groin crease, posY≈0.06 — check whether N1 (deep vein) and N2 (GSV) "
            "are both visible at groin level to confirm or exclude SFJ incompetence."
        )
    else:
        lines.append(
            "Q1 NOT CONFIRMED — Move probe to groin crease, anterior surface, posY≈0.06. "
            "VLM should show N2 (GSV) in fascial compartment alongside N1 (deep vein) at SFJ. "
            "If SFJ competent, scan medial thigh for Hunterian perforator (posY 0.21–0.35)."
        )

    # ── Q2: GSV trunk reflux (required when Q1 = SFJ/Hunterian or perforator) ─
    if has_ep_n1_n2 or has_ep_n2_n2:
        ep_clip = first("EP", "N1", "N2") or first("EP", "N2", "N2")
        ep_pos  = ep_clip.get("posYRatio", 0.06) if ep_clip else 0.06

        if has_rp_n2_n1:
            c = first("RP", "N2", "N1")
            lines.append(
                f"Q2 CONFIRMED — RP N2→N1: GSV trunk reflux at {_posY_to_level(c.get('posYRatio', 0.3))}. "
                "Trunk is carrying blood backward (toward foot)."
            )
        else:
            lines.append(
                f"Q2 NOT CONFIRMED — Scan GSV trunk distally from posY≈{ep_pos:.2f}, "
                "anterior medial thigh surface. "
                "VLM should show N2 continuously in fascial compartment along trunk. "
                "If retrograde flow seen: RP N2→N1. If no trunk reflux: GSV may be conduit to tributary escape (Q3)."
            )

    # ── Q3: Tributary escape (required when Q1 = SFJ/Hunterian or perforator) ─
    if has_ep_n1_n2 or has_ep_n2_n2:
        if has_ep_n2_n3:
            c = first("EP", "N2", "N3")
            lines.append(
                f"Q3 CONFIRMED — EP N2→N3: Blood escapes from GSV into tributary at "
                f"{_posY_to_level(c.get('posYRatio', 0.3))} ({c.get('legSide','?')} leg). "
                "VLM at this posY should show both N2 in fascia AND N3 above fascia."
            )
        else:
            # Give location hint based on what's confirmed
            if has_rp_n2_n1:
                rp = first("RP", "N2", "N1")
                rp_pos = rp.get("posYRatio", 0.3) if rp else 0.3
                lines.append(
                    f"Q3 NOT CONFIRMED — Scan along GSV between posY≈0.06 (SFJ) and posY≈{rp_pos:.2f} (trunk reflux level). "
                    "Watch for N3 appearing above fascia while N2 is in compartment — that junction is the escape point (EP N2→N3). "
                    "Note: if GSV conduit to escape point, RP N2→N1 may NOT be present below escape (Type 3 rule)."
                )
            else:
                lines.append(
                    "Q3 NOT CONFIRMED — After confirming Q2, scan distally along GSV. "
                    "When VLM shows N3 above fascia at same level as N2 in fascia, that is the likely escape point (EP N2→N3)."
                )

    # ── Q3 for perforator entry (EP N2→N2): look for RP N3 near perforator ──
    if has_ep_n2_n2 and not has_ep_n1_n2:
        c = first("EP", "N2", "N2")
        perf_pos = c.get("posYRatio", 0.25) if c else 0.25
        if not has_rp_n3:
            lines.append(
                f"PERFORATOR CIRCUIT — EP N2→N2 at posY≈{perf_pos:.2f}. "
                f"Look for tributary reflux (RP N3) near same level: scan for N3 above fascia at posY "
                f"{max(0.0, perf_pos-0.10):.2f}–{min(1.0, perf_pos+0.15):.2f}. "
                "Also check if GSV trunk refluxes below perforator (RP N2→N1) — distinguishes Type 2B from Type 2C."
            )
        elif has_rp_n3 and not has_rp_n2_n1:
            lines.append(
                "PERFORATOR CIRCUIT (Type 2B so far) — RP N3 confirmed. "
                "Check for RP N2→N1 (GSV trunk reflux) to distinguish Type 2B (no trunk reflux) from Type 2C (trunk also refluxes)."
            )

    # ── Q4: Tributary reflux (required when Q3 = confirmed) ──────────────────
    if has_ep_n2_n3:
        escape = first("EP", "N2", "N3")
        esc_pos = escape.get("posYRatio", 0.3) if escape else 0.3

        if has_rp_n3:
            rp3 = first_rp_n3()
            direction = "back toward GSV (RP N3→N2)" if has_rp_n3_n2 else "into deep system (RP N3→N1)"
            rp3_level = _posY_to_level(rp3.get("posYRatio", 0.5)) if rp3 else "unknown level"
            lines.append(
                f"Q4 CONFIRMED — RP N3 confirmed: tributary reflux {direction} at {rp3_level}."
            )
        else:
            lines.append(
                f"Q4 NOT CONFIRMED — Follow tributary branch from escape point at posY≈{esc_pos:.2f}. "
                "VLM should show N3 above fascia. Scan along branch direction — "
                "retrograde flow toward GSV = RP N3→N2; retrograde into deep system via perforator = RP N3→N1."
            )

    # ── EP N1→N3 circuit: Type 4/5 branch ────────────────────────────────────
    if has_ep_n1_n3:
        if has_rp_n2_n1 and not has_rp_n3:
            c = first("EP", "N1", "N3")
            lines.append(
                f"TYPE 4 PATTERN developing — EP N1→N3 at {_posY_to_level(c.get('posYRatio', 0.25))} + RP N2→N1. "
                "Still need to confirm: does N3 drain into GSV (RP N3→N2) before GSV returns to deep? "
                "Scan the tributary for backward flow toward GSV."
            )
        elif not has_rp_n2_n1 and not has_rp_n3:
            c = first("EP", "N1", "N3")
            lines.append(
                f"TYPE 4/5 PATTERN — EP N1→N3 confirmed at {_posY_to_level(c.get('posYRatio', 0.25))}. "
                "Need to know how blood re-enters deep system: "
                "via GSV trunk (RP N3→N2 then RP N2→N1 = Type 4), or directly via perforator (RP N3→N1 = Type 5). "
                "Scan tributary for backward flow."
            )

    # ── Type 3 conduit check ──────────────────────────────────────────────────
    if has_ep_n1_n2 and has_ep_n2_n3 and not has_rp_n2_n1:
        escape = first("EP", "N2", "N3")
        esc_pos = escape.get("posYRatio", 0.3) if escape else 0.3
        lines.append(
            f"TYPE 3 CONDUIT CHECK — EP N1→N2 + EP N2→N3 confirmed, NO RP N2→N1 yet. "
            f"GSV may be acting as conduit from SFJ to escape at posY≈{esc_pos:.2f} (Type 3 pattern). "
            "Do NOT assume trunk reflux below escape point — only record RP N2→N1 if explicitly seen below escape posY."
        )

    return "\n".join(lines) if lines else "No status determined."


@dataclass
class ActiveGuidanceResponse:
    location: dict = field(default_factory=dict)
    guidance: str = ""
    ultrasound_assessment: Optional[dict] = None
    expected_region: Optional[str] = None
    error: Optional[str] = None
    llm_system: str = ""
    llm_prompt: str = ""
    llm_raw: str = ""

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "guidance": self.guidance,
            "ultrasound_assessment": self.ultrasound_assessment,
            "expected_region": self.expected_region,
            "error": self.error,
            "debug": {
                "llm_system": self.llm_system,
                "llm_prompt": self.llm_prompt,
                "llm_raw": self.llm_raw,
            },
        }


def _build_prompt(
    probe: ProbeLocation,
    wrong: WrongRegionResult,
    expected_region: Optional[str],
    clips: list[dict],
    vlm_summary: str,
) -> str:
    pct  = f"{probe.segment_dist * 100:.0f}%"
    side = "anterior" if probe.is_front else "posterior"

    lines = [
        "## Current probe position",
        f"Region: {probe.region} ({probe.region_full_name})",
        f"Leg: {probe.leg}  |  Surface: {side}  |  {pct} along segment ({_posY_to_level(probe.segment_dist)})",
    ]

    if expected_region:
        if wrong.is_wrong:
            lines.append(f"Expected region: {expected_region} — WRONG REGION. {wrong.suggestion or ''}")
        else:
            lines.append(f"Expected region: {expected_region} — correctly placed.")

    lines += [
        "",
        "## Confirmed EP/RP clips",
        _ep_rp_summary(clips),
        "",
        "## CHIVA Q1-Q4 status + circuit prediction",
        _chiva_component_status(clips),
        "",
        "## VLM frame: annotated anatomy visible right now",
        vlm_summary,
        "",
        "## Task: ONE line, ≤12 words — the single next action.",
        'JSON only: {"guidance": "<≤12 words>"}',
    ]
    return "\n".join(lines)


def _call_groq(
    system: str,
    prompt: str,
    api_key: str,
    model: str,
    history: list[dict] | None = None,
) -> tuple[dict, str]:
    from groq import Groq
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": system},
        *(history or []),
        {"role": "user",   "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=model,
        max_tokens=40,
        temperature=0.0,
        messages=messages,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw_original = raw
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```"))
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1:
        return json.loads(raw[s:e+1]), raw_original
    raise ValueError(f"LLM returned non-JSON: {raw_original[:200]}")


def generate_guidance(
    probe: ProbeLocation,
    expected_region: Optional[str] = None,
    ep_rp_findings: Optional[dict] = None,
    vlm_assessment: Optional[UltrasoundAssessment] = None,
    current_clip_index: int = 0,
    session_id: str = "default",
) -> ActiveGuidanceResponse:
    from config import GROQ_API_KEY, GROQ_TEXT_MODEL

    wrong = check_wrong_region(probe.region, expected_region)
    clips: list[dict] = (ep_rp_findings or {}).get("clips", [])

    prompt = _build_prompt(
        probe, wrong, expected_region,
        clips,
        vlm_assessment.summary() if vlm_assessment else "No frame provided.",
    )

    history = _get_history(session_id)

    resp = ActiveGuidanceResponse(
        location=probe.to_dict(),
        expected_region=expected_region,
        llm_system=_CHIVA_SYSTEM_PROMPT,
        llm_prompt=prompt,
    )

    try:
        result, raw = _call_groq(
            _CHIVA_SYSTEM_PROMPT, prompt, GROQ_API_KEY, GROQ_TEXT_MODEL,
            history=history,
        )
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        resp.error = str(exc)
        resp.guidance = f"LLM error: {exc}"
        resp.ultrasound_assessment = vlm_assessment.to_dict() if vlm_assessment else None
        return resp

    resp.guidance = result.get("guidance", "")
    resp.llm_raw = raw
    resp.ultrasound_assessment = vlm_assessment.to_dict() if vlm_assessment else None

    _push_history(session_id, prompt, raw)
    return resp