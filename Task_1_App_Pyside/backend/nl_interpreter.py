"""
Natural Language to CHIVA Notation Interpreter
-----------------------------------------------
Converts a surgeon's natural language description of venous blood flow
into CHIVA virtual clips that can be passed directly to the classification
and ligation planning pipeline.
"""

import json
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

_NL_TO_CHIVA_PROMPT_OLD = """You are an expert CHIVA vascular surgeon. A colleague is describing a patient's venous flow condition in plain clinical language. Your job is to translate that description into CHIVA clip notation so the AI classification system can process it.

=== CHIVA NOTATION GUIDE ===

COMPARTMENTS:
  N1 = Deep venous system (femoral vein, popliteal vein, deep veins)
  N2 = Saphenous trunk (GSV = Great Saphenous Vein, SSV = Small Saphenous Vein)
  N3 = Tributaries, perforators branching off the saphenous trunk, varicosities, superficial branches

FLOW DIRECTIONS:
  EP = Antegrade (forward, physiological, normal direction toward heart)
  RP = Retrograde (reflux, backward, pathological, away from heart)

POSITION RATIOS (posYRatio — 0 = groin, 1 = ankle):
  SFJ / groin area:           0.04 – 0.09
  Upper thigh:                0.10 – 0.20
  Mid thigh (Hunterian area): 0.21 – 0.35
  Knee / popliteal area:      0.40 – 0.55
  Calf:                       0.60 – 0.80
  Ankle:                      0.85 – 1.00

KEY MAPPINGS FROM CLINICAL LANGUAGE:
  "SFJ incompetent" / "reflux at SFJ" / "deep blood enters GSV at groin"
      → EP  N1→N2  posYRatio≈0.06

  "GSV reflux" / "blood flows backward in GSV" / "GSV carries reflux downward"
      → RP  N2→N1  (at the level described, e.g. mid-thigh ≈ 0.30)

  "blood escapes to tributaries" / "GSV feeds tributaries" / "EP from GSV to branch" /
  "GSV feeds a tributary" / "GSV discharges into a tributary" / "discharges forward into a tributary" /
  "GSV discharges blood into" / "blood exits the GSV into a tributary" / "GSV empties into a tributary"
      → EP  N2→N3  (at the level described)

  "blood refluxes back in the tributary" / "tributary drains backward"
      → RP  N3→N2 or RP N3→N1  (depending on where it drains to)

  "perforator feeds GSV" / "there is a perforator entry into the trunk"
      → EP  N2→N2  (perforator entry — note: fromType=N2, toType=N2, NOT N1→N2)
      This means SFJ is COMPETENT even if posYRatio is small.

  "deep vein directly feeds a tributary" / "N1 to N3 direct connection"
      → EP  N1→N3

  "Hunterian perforator incompetent" / "mid-thigh perforator entry from deep system"
      → EP  N1→N2  posYRatio≈0.25 (SFJ INCOMPETENT via Hunterian)

IMPORTANT DISTINCTIONS:
  - EP N1→N2 means deep system → saphenous trunk (SFJ or Hunterian INCOMPETENT)
  - EP N2→N2 means perforator → saphenous trunk (SFJ COMPETENT)
  - RP N2→N1 means saphenous trunk reflux (backward in GSV)
  - RP N3→N2 means tributary reflux back into GSV
  - RP N3→N1 means tributary reflux all the way back to deep system

=== CLINICAL DESCRIPTION TO INTERPRET ===
{description}

=== INSTRUCTIONS ===
1. Read the description carefully.
2. Identify each distinct blood flow event mentioned.
3. Generate one virtual clip per flow event.
4. Use the mappings above to assign EP/RP and N1/N2/N3 notation.
5. Estimate posYRatio from anatomical location clues.
6. If left/right leg is explicitly mentioned, assign legSide accordingly. If NOT mentioned, use "Unspecified" — never assume or default to Left or Right.
7. If the description is NOT about patient venous anatomy (e.g., it is a question,
   a greeting, or asks about a concept without describing a patient), set is_clinical=false.

Output ONLY valid JSON — no markdown, no explanation:
{{
    "is_clinical": true,
    "interpretation": "<2-3 sentences summarising the findings in CHIVA terms>",
    "clips": [
        {{
            "flow": "EP",
            "fromType": "N1",
            "toType": "N2",
            "posYRatio": 0.06,
            "step": "SFJ",
            "legSide": "Unspecified"
        }}
    ]
}}

If not clinical:
{{
    "is_clinical": false,
    "interpretation": null,
    "clips": []
}}"""

_NL_TO_CHIVA_PROMPT = """You are an expert CHIVA vascular surgeon. A colleague is describing a patient's venous flow condition in plain clinical language. Your job is to translate that description into CHIVA clip notation so the AI classification system can process it.

=== CHIVA NOTATION GUIDE ===

COMPARTMENTS — READ CAREFULLY:
  N1 = Deep venous system ONLY (femoral vein, popliteal vein, deep veins — the actual named deep vessel)
  N2 = Saphenous TRUNK ONLY — i.e. the GSV (Great Saphenous Vein) or SSV (Small Saphenous Vein) named explicitly.
       N2 applies ONLY when the clinician specifically names the GSV, SSV, or saphenous trunk.
  N3 = Everything else superficial: tributaries, branches, varicosities, perforators,
       AND generic "superficial veins" / "superficial system" when the specific trunk is NOT named.

  *** CRITICAL: "superficial veins" without naming GSV/SSV = N3, NOT N2. ***
  *** N2 is reserved for the named saphenous trunk (GSV or SSV) only. ***

FLOW DIRECTIONS:
  EP = Antegrade (forward, physiological, normal direction toward heart)
  RP = Retrograde (reflux, backward, pathological, away from heart)

POSITION RATIOS (posYRatio — 0 = groin, 1 = ankle):
  SFJ / groin area:           0.04 – 0.09
  Upper thigh:                0.10 – 0.20
  Mid thigh (Hunterian area): 0.21 – 0.35
  Knee / popliteal area:      0.40 – 0.55
  Calf:                       0.60 – 0.80
  Ankle:                      0.85 – 1.00

═══════════════════════════════════════════════════════════
CRITICAL RULE 1 — EP N2→N2 vs EP N1→N2 (most common error):
═══════════════════════════════════════════════════════════

  EP N2→N2  (fromType=N2, toType=N2) — SFJ COMPETENT, perforator entry:
    Use when a perforating vessel inserts into, enters, or feeds the GSV — UNLESS the
    description explicitly states the perforator bridges the DEEP VENOUS SYSTEM to the GSV
    (see Hunterian Exception below).
    The descriptor "deep" alone (e.g. "deep perforating vessel") describes anatomical depth
    only — it does NOT mean N1 origin unless the deep venous system is explicitly the blood source.
    Trigger phrases: "perforator enters the GSV", "perforator inserts into the GSV",
    "perforating vessel connects to the saphenous trunk", "perforator feeds mid-GSV",
    "communicating vein inserts into the GSV", "SFJ competent but a perforator connects to the GSV".

  EP N1→N2  (fromType=N1, toType=N2) — SFJ or Hunterian INCOMPETENT:
    Use when the DEEP VENOUS SYSTEM ITSELF delivers blood into the GSV, either at the SFJ
    (groin) or via an incompetent Hunterian perforator (upper/mid-thigh).
    Trigger phrases: "SFJ incompetent", "femoral vein feeds GSV", "deep venous blood enters GSV",
    "blood from deep system enters GSV", "Hunterian perforator incompetent".

  *** HUNTERIAN EXCEPTION — EP N1→N2 even when the word "perforator" is present: ***
    This exception has a VERY NARROW trigger. Use EP N1→N2 ONLY when the description
    uses one of these specific phrases:
      ✓ "[perforator] connects the deep system to the GSV"     → EP N1→N2
      ✓ "Hunterian perforator incompetent"                     → EP N1→N2
      ✓ "incompetent Hunterian"                                → EP N1→N2
      ✓ "allowing deep venous blood to enter the GSV directly" → EP N1→N2

    The following phrases look similar but are NOT Hunterian — use EP N2→N2:
      ✗ "from the deep system"         → EP N2→N2 (anatomical position, not the blood source driving the shunt)
      ✗ "from a deep perforating vessel" → EP N2→N2 ("deep" describes the vessel's depth, NOT that N1 is the source)
      ✗ "perforator entry from the deep system" → EP N2→N2 (positional phrase, not a drive from N1)
      ✗ "a deep perforating vessel enters the GSV" → EP N2→N2 ("deep" = anatomical depth only)
      ✗ "perforator at [location] enters/inserts into/feeds the GSV" → EP N2→N2
      ✗ "a perforating vessel feeds into the GSV from the deep system" → EP N2→N2
         (even though "from the deep system" is present — "perforating vessel feeds GSV" is a perforator entry, NOT N1→N2)
      ✗ "feeds into the GSV from the deep system" → EP N2→N2 (perforator = N2→N2 regardless of depth qualifier)

    CRITICAL DISTINCTION:
      "connects the deep SYSTEM to the GSV" (deep system = the blood driver) → EP N1→N2
      "from the deep system" / "from a deep vessel" (positional or descriptive) → EP N2→N2

  DECISION RULE:
    Does the description use the exact phrase "connects the deep system to the GSV"
    OR explicitly call the Hunterian perforator incompetent?
      YES → EP N1→N2 (Hunterian incompetent — even if SFJ stated as competent)
      NO  → EP N2→N2 (perforator entry, SFJ competent)

═══════════════════════════════════════════════════════════
CRITICAL RULE 2 — DO NOT HALLUCINATE RP CLIPS:
═══════════════════════════════════════════════════════════

  Generate RP clips ONLY when the description EXPLICITLY contains words such as:
    "reflux", "refluxes", "retrograde", "backward", "backwards", "back toward the deep",
    "drains back", "flows back", "carries blood back", or direct equivalent phrasing.

  If the description mentions ONLY forward / antegrade flow — words like "feeds", "enters",
  "passes into", "escapes to", "discharges into", "flows forward", "antegrade" — and contains
  NO mention of backward or retrograde flow anywhere, then generate EP clips ONLY.

  NEGATIVE STATEMENTS CONFIRM NO RP — treat these as zero RP when global in scope:
    "no retrograde flow detected anywhere", "no reflux anywhere", "no reflux is present anywhere",
    "no retrograde flow is identified", "not refluxing anywhere", "without any reflux" — when
    these phrases appear and clearly deny ALL reflux (not just reflux in one specific vessel),
    generate ZERO RP findings regardless of what EP pattern is described.
    *** IMPORTANT: "no reflux in the GSV trunk" or "GSV is not refluxing" are PARTIAL statements
    about one vessel — they do NOT mean zero RP overall. Do not suppress RP N3 findings because
    the GSV trunk specifically has no reflux. ***

  SFJ INCOMPETENCE ALONE DOES NOT IMPLY RP:
    The presence of SFJ incompetence (EP N1→N2) does NOT automatically mean GSV reflux
    (RP N2→N1) exists. They are separate findings. Only add RP N2→N1 if the description
    explicitly states the GSV is refluxing or carrying blood backward.

  CONCRETE NO-SHUNT EXAMPLES — generate EP clips only, zero RP:
    "SFJ incompetent, blood enters GSV at groin, no retrograde flow detected anywhere"
        → [EP N1→N2]  — zero RP clips
    "A perforating vessel enters the GSV, SFJ competent, no reflux present"
        → [EP N2→N2]  — zero RP clips
    "Blood enters GSV at SFJ and feeds a tributary, no backward flow identified"
        → [EP N1→N2, EP N2→N3]  — zero RP clips

  *** DO NOT infer or add RP findings that are not explicitly stated in the description. ***
  *** Negative reflux statements (global scope) = zero RP. SFJ entry alone ≠ GSV reflux. ***
  *** Never fabricate reflux to make the pattern fit a known shunt type. ***

CRITICAL RULE 2B — eliminationTest values and when to add:
  Only add "eliminationTest" to a finding if the description EXPLICITLY describes performing
  a compression or elimination test AND states its result (e.g. "reflux persists", "reflux
  disappears", "elimination test positive/negative", "compression abolished reflux").
  If no elimination test is mentioned → do NOT add eliminationTest to any finding.
  *** Never infer or guess an eliminationTest result from the reflux pattern alone. ***

  eliminationTest VALUE ASSIGNMENT:
    Use "Reflux" when compression ABOLISHED / ELIMINATED tributary reflux:
      Trigger phrases: "reflux disappeared on compression", "reflux abolished", "tributary reflux gone",
      "compression eliminated all reflux", "reflux ceased", "tributary reflux disappeared",
      "elimination test positive", "reflux disappeared", "reflux eliminated on compression"
      → eliminationTest="Reflux"  (Type 1+2 pattern confirmed — GSV is the single feeder)

    Use "No Reflux" when reflux PERSISTED / was UNCHANGED despite compression:
      Trigger phrases: "reflux persists", "reflux unchanged", "no change on compression",
      "tributary continued to reflux", "reflux not abolished"
      → eliminationTest="No Reflux"  (Type 3 pattern — tributary has independent source)

═══════════════════════════════════════════════════════════

KEY MAPPINGS AND CLUES FOR EP/RP IDENTIFICATION:
  "SFJ incompetent" / "reflux at SFJ" / "deep blood enters GSV at groin"
      → EP  N1→N2

  "GSV reflux" / "blood flows backward in GSV" / "GSV carries reflux downward" /
  "full-length GSV reflux" / "GSV refluxes throughout" / "GSV is incompetent and refluxes"
      → RP  N2→N1  (at the level described, e.g. mid-thigh ≈ 0.30)
      *** Always generate RP N2→N1 as a SEPARATE clip from EP N1→N2.
          EP N1→N2 = shunt entry. RP N2→N1 = GSV trunk reflux. Both can and DO coexist. ***

  "blood escapes to tributaries" / "GSV feeds tributaries" / "EP from GSV to branch" /
  "GSV feeds a tributary" / "GSV discharges into a tributary" / "discharges forward into a tributary" /
  "GSV discharges blood into" / "blood exits the GSV into a tributary" / "GSV empties into a tributary"
      → EP  N2→N3  (at the level described)

  "blood refluxes back in the tributary" / "tributary drains backward" / "tributary shows retrograde flow"
  / "tributary carries blood backward" / "tributary refluxes back toward the GSV"
      → RP  N3→N2  (tributary carrying blood backward toward the GSV)
      (If the GSV discharging INTO the tributary was also described separately, also generate EP N2→N3)

  "tributary drains backward into the deep system via perforator" / "tributary re-enters the deep system" /
  "tributary connects back to deep vein" / "tributary drains to deep via perforating vessel"
      → RP  N3→N1  (tributary draining all the way to deep system via perforator)
      (If the GSV discharging INTO the tributary was also described separately, also generate EP N2→N3)

  "perforator feeds GSV" / "perforating vessel enters the GSV" / "perforator inserts into the trunk"
      → EP  N2→N2  (perforator entry — fromType=N2, toType=N2, SFJ COMPETENT)

  "deep vein to superficial veins" / "deep to superficial" (GSV NOT named)
      → EP  N1→N3

  "superficial veins reflux back to deep" / "superficial back to deep vein" (GSV NOT named)
      → RP  N3→N1

  "deep vein directly feeds a tributary" / "N1 to N3 direct connection"
      → EP  N1→N3

  "Hunterian perforator incompetent" / "mid-thigh perforator entry from deep system allows deep blood into GSV"
      → EP  N1→N2 (SFJ INCOMPETENT via Hunterian — only when explicitly stated as incompetent)

IMPORTANT DISTINCTIONS:
  - EP N1→N2 means deep system → named saphenous trunk GSV/SSV (SFJ or Hunterian INCOMPETENT)
  - EP N1→N3 means deep system → generic superficial veins/tributaries (when GSV/SSV NOT named)
  - EP N2→N2 means perforator/communicating vein → saphenous trunk (SFJ COMPETENT)
  - RP N2→N1 means saphenous trunk reflux (backward in GSV)
  - RP N3→N2 means tributary reflux back into GSV
  - RP N3→N1 means tributary/superficial reflux all the way back to deep system

=== CLINICAL DESCRIPTION TO INTERPRET ===
{description}

=== INSTRUCTIONS ===

─── STEP 1: SUFFICIENCY GATE (decide this before generating any clips) ───

CRITICAL RULE: "X is incompetent" or "X incompetent" — even combined with anatomical
location words — is a STRUCTURAL STATE LABEL, not a blood-movement description.
It tells you a valve is leaky but says NOTHING about what blood actually does.
These ALWAYS get sufficient_information = false, no matter what other anatomy words surround them.

sufficient_information = false — ALWAYS reject inputs like these:
  ✗ "SFJ incompetent"
  ✗ "SSV SFJ incompetent Groin"
  ✗ "SSV SFJ incompetent Groin at Saphenous Trunk"
  ✗ "Mid thigh GSV SSV SFJ incompetent Saphenous Trunk"
  ✗ "GSV SFJ N1 N4 N5 N20"
  ✗ "patient has varicose veins"
  ✗ "leg swelling"
  ✗ "SFJ incompetent, mid-thigh perforator" (state labels + anatomy, still no flow event)

The pattern: if the whole input is anatomy names / state labels / locations with no verb
describing blood MOVEMENT — it is insufficient.

sufficient_information = true — only when blood movement is explicitly described:
  ✓ "There is reflux at the SFJ flowing into the GSV down to mid-thigh"
  ✓ "SFJ incompetent, GSV refluxes full-length to the knee"  ← "refluxes" = flow verb
  ✓ "Blood enters GSV at groin, discharges into a calf tributary which then refluxes back"
  ✓ "Perforator feeds the GSV at mid-thigh, no reflux present"  ← "feeds" = flow verb

The test: can you complete the sentence "blood is ___ing through ___"? If yes → sufficient.
If all you can say is "the SFJ is incompetent / is present / is at the groin" → insufficient.

If sufficient_information = false:
  → Set clips = [], interpretation = null
  → Set missing_information = a specific, friendly sentence explaining what duplex
    data is needed (e.g. "Please describe what the blood is doing — for example,
    whether the GSV is refluxing, where it exits into tributaries, and whether the
    SFJ or any perforators are feeding the system.")
  → Output the JSON immediately — DO NOT proceed to clip generation

If sufficient_information = true → continue to STEP 2.

─── STEP 2: CLIP GENERATION (only if STEP 1 passed) ───

1. Read the description carefully.
2. Identify each distinct blood flow event EXPLICITLY mentioned.
3. Generate one virtual clip per flow event.
4. Use the mappings above to assign EP/RP and N1/N2/N3 notation.
5. Estimate posYRatio from anatomical location clues.
6. If left/right leg is explicitly mentioned, assign legSide accordingly. If NOT mentioned, use "Unspecified" — never assume or default to Left or Right.
7. If the description is NOT about patient venous anatomy (e.g., it is a question,
   a greeting, or asks about a concept without describing a patient), set is_clinical=false.
8. Apply Critical Rule 1: perforator entering GSV = EP N2→N2, UNLESS the description uses the
   exact phrase "connects the deep system to the GSV" or calls the Hunterian perforator incompetent.
   "from the deep system" or "from a deep perforating vessel" = anatomical description → EP N2→N2.
9. Apply Critical Rule 2: generate RP findings ONLY when backward/retrograde/reflux is explicitly stated.
   "No reflux in [specific vessel]" is a partial statement — do NOT suppress RP findings in other vessels.
10. Apply Critical Rule 2B: do NOT add eliminationTest unless the description explicitly describes
    performing a compression/elimination test and states its result. Use "Reflux" when compression
    ABOLISHED tributary reflux (confirming GSV is the source). Use "No Reflux" when reflux PERSISTED.
11. CRITICAL — RP N2→N1 is separate from EP N1→N2: When "GSV refluxes" / "full-length GSV reflux" /
    "GSV carries blood backward" is EXPLICITLY stated, always generate RP N2→N1 as a separate clip
    in addition to EP N1→N2. These are two different flow events at two different clip positions.

Output ONLY valid JSON — no markdown, no explanation:
{{
    "is_clinical": true,
    "sufficient_information": true,
    "missing_information": null,
    "interpretation": "<2-3 sentences summarising the findings in CHIVA terms>",
    "clips": [
        {{
            "flow": "EP",
            "fromType": "N1",
            "toType": "N2",
            "posYRatio": 0.06,
            "step": "SFJ",
            "legSide": "Unspecified"
        }}
    ]
}}

If not clinical or insufficient information:
{{
    "is_clinical": true,
    "sufficient_information": false,
    "missing_information": "<specific sentence about what duplex data is needed>",
    "interpretation": null,
    "clips": []
}}

If not clinical at all (question, greeting, concept):
{{
    "is_clinical": false,
    "sufficient_information": false,
    "missing_information": null,
    "interpretation": null,
    "clips": []
}}"""

_CONVERSATIONAL_PROMPT = """You are a knowledgeable CHIVA vascular surgery assistant helping a clinician understand venous shunt classification and CHIVA treatment decisions.

The clinician understands anatomy and vascular surgery in general, but may have limited knowledge of the specific CHIVA hemodynamic classification system.

=== PRIOR CLINICAL ANALYSIS ===
{analysis_context}

=== CONVERSATION HISTORY ===
{history}

=== CLINICIAN'S QUESTION ===
{user_message}

Respond clearly and concisely. Use clinical language appropriate for a vascular surgeon.
Reference the specific findings from the analysis when relevant.
Keep the answer focused — avoid unnecessary preamble."""


_NO_REFLUX_PHRASES = [
    # Only match phrases that unambiguously deny reflux GLOBALLY (not vessel-specific statements).
    # Bare "no reflux" / "no retrograde flow" are intentionally excluded — they match partial
    # vessel-specific sentences like "no reflux in the GSV trunk" which do NOT mean zero RP overall.
    "no retrograde flow is detected anywhere",
    "no retrograde flow detected anywhere",
    "no retrograde flow identified anywhere",
    "no retrograde flow is identified anywhere",
    "no retrograde flow anywhere",
    "no retrograde flow is identified",
    "no retrograde flow is detected",
    "no reflux anywhere",
    "no reflux is present anywhere",
    "no reflux detected anywhere",
    "no reflux identified anywhere",
    "no reflux is identified",
    "no backward flow anywhere",
    "no backward flow is detected anywhere",
    "no backward flow detected anywhere",
    "no backward flow identified anywhere",
    "no backward flow is identified anywhere",
    "no backward flow is detected",
    "backward flow is not detected",
    "not refluxing anywhere",
    "without any reflux",
    "reflux-free",
    "no pathological reflux",
    "reflux is absent",
    "reflux is not present",
    "no reflux is present",
    "no reflux identified in the gsv or tributaries",
    "no reflux in the gsv or tributaries",
]


def _has_no_reflux_statement(description: str) -> bool:
    desc_lower = description.lower()
    return any(phrase in desc_lower for phrase in _NO_REFLUX_PHRASES)


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def parse_nl_to_clips(user_message: str, call_llm_fn: Callable) -> dict:
    """
    Ask the LLM to convert a natural-language description to CHIVA clip notation.

    Returns:
        {
            "is_clinical": bool,
            "sufficient_information": bool,
            "missing_information": str | None,
            "interpretation": str | None,
            "clips": list[dict]
        }
    """
    prompt = _NL_TO_CHIVA_PROMPT.format(description=user_message.strip())
    try:
        raw, _ = call_llm_fn(prompt, return_usage=True, max_tokens=1024)
        raw = _clean_json(raw)
        result = json.loads(raw)
        if isinstance(result, dict) and "is_clinical" in result and "clips" in result:
            # Deterministic post-process: strip hallucinated RP clips when description
            # explicitly states no reflux/retrograde flow. LLM training data strongly
            # associates SFJ incompetence with GSV reflux and overrides prompt rules.
            if _has_no_reflux_statement(user_message) and result.get("clips"):
                before = len(result["clips"])
                result["clips"] = [c for c in result["clips"] if c.get("flow") != "RP"]
                after = len(result["clips"])
                if before != after:
                    logger.info(
                        f"Post-processing stripped {before - after} hallucinated RP clip(s) "
                        f"because description contains explicit no-reflux statement."
                    )
            # Default sufficient_information to True if LLM omitted the field (backwards compat)
            if "sufficient_information" not in result:
                result["sufficient_information"] = bool(result.get("clips"))
            return result
    except Exception as e:
        logger.error(f"NL interpretation failed: {e}")
    return {"is_clinical": False, "sufficient_information": False, "missing_information": None, "clips": []}


def build_conversational_response(
    user_message: str,
    analysis_context: str,
    history: list[dict],
    call_llm_fn: Callable,
) -> str:
    """Generate a conversational follow-up response given the analysis context."""
    history_lines = []
    for m in history[-8:]:
        role_label = "Clinician" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "")[:400]
        history_lines.append(f"{role_label}: {content}")

    prompt = _CONVERSATIONAL_PROMPT.format(
        analysis_context=analysis_context or "No prior clinical analysis available.",
        history="\n".join(history_lines) or "(start of conversation)",
        user_message=user_message.strip(),
    )
    try:
        response, _ = call_llm_fn(prompt, return_usage=True, max_tokens=900)
        return response.strip()
    except Exception as e:
        logger.error(f"Conversational response failed: {e}")
        return (
            "I'm sorry, I encountered an error generating a response. "
            "Please check your network connection and try again."
        )
