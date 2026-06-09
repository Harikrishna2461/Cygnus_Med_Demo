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

═══════════════════════════════════════════════════════════
QUICK REFERENCE — WHAT EACH SHUNT TYPE REQUIRES:
═══════════════════════════════════════════════════════════
Use this to understand what flow information makes a complete description.

  TYPE 1   = EP N1→N2  +  RP N2→N1
             (SFJ/Hunterian entry into GSV  +  GSV trunk refluxes backward)

  TYPE 2A  = EP N2→N3  only  (no RP required)
             (GSV overflows forward into tributary; no reflux established yet)

  TYPE 2B  = EP N2→N2  +  RP N3→N2 or RP N3→N1  (no RP N2→N1)
             (perforator feeds GSV mid-segment + tributary reflux; trunk does NOT reflux)

  TYPE 2C  = EP N2→N2  +  RP N3  +  RP N2→N1
             (perforator entry + tributary reflux + GSV trunk also refluxes backward)

  TYPE 3   = EP N1→N2  +  EP N2→N3  +  RP N3  (no RP N2→N1)
             (SFJ entry + GSV escapes to tributary + tributary refluxes; trunk does NOT reflux)

  TYPE 4   = EP N1→N3  +  RP N2→N1
             (deep blood enters tributary directly, bypassing GSV + GSV trunk refluxes back)

  TYPE 5   = EP N1→N3  +  RP N3→N1 or RP N3→N2
             (deep blood enters tributary directly + reflux stays within tributaries)

  TYPE 1+2 = EP N1→N2  +  EP N2→N3  +  RP N3  +  RP N2→N1  +  eliminationTest result
             (dual entry at SFJ and tributary + both trunk and tributary reflux + test)

  NO SHUNT = No RP findings anywhere
             (all flow antegrade; sole exception is Type 2A which has EP N2→N3 only)
═══════════════════════════════════════════════════════════

=== CLINICAL DESCRIPTION TO INTERPRET ===
{description}


=== INSTRUCTIONS ===
1. Read the description carefully.
2. Identify each distinct blood flow event EXPLICITLY mentioned.
3. Generate one virtual clip per flow event.
4. Use the mappings above to assign EP/RP and N1/N2/N3 notation.
5. Estimate posYRatio from anatomical location clues.
6. If left/right leg is explicitly mentioned, assign legSide accordingly. If NOT mentioned, use "Unspecified" — never assume or default to Left or Right.
7. Apply Critical Rule 1: perforator entering GSV = EP N2→N2, UNLESS the description uses the
   exact phrase "connects the deep system to the GSV" or calls the Hunterian perforator incompetent.
   "from the deep system" or "from a deep perforating vessel" = anatomical description → EP N2→N2.
8. Apply Critical Rule 2: generate RP findings ONLY when backward/retrograde/reflux is explicitly stated.
   "No reflux in [specific vessel]" is a partial statement — do NOT suppress RP findings in other vessels.
9. Apply Critical Rule 2B: do NOT add eliminationTest unless the description explicitly describes
   performing a compression/elimination test and states its result. Use "Reflux" when compression
   ABOLISHED tributary reflux (confirming GSV is the source). Use "No Reflux" when reflux PERSISTED.
10. CRITICAL — RP N2→N1 is separate from EP N1→N2: When "GSV refluxes" / "full-length GSV reflux" /
    "GSV carries blood backward" is EXPLICITLY stated, always generate RP N2→N1 as a separate clip
    in addition to EP N1→N2. These are two different flow events at two different clip positions.

Output ONLY valid JSON — no markdown, no explanation:
{{
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
}}"""

_SUFFICIENCY_PROMPT = """You are a strict clinical gatekeeper for a CHIVA venous shunt classification tool. Your ONLY job is to decide whether the accumulated description below contains enough explicitly confirmed information to classify the shunt — nothing more.

The messages below are numbered chronologically — [Message 1] is the earliest, the highest-numbered message is the most recent.

PRECEDENCE RULE: When later messages contradict earlier ones about the same component, the LATER message is the authoritative answer. Do NOT keep re-asking about something the clinician already answered, even if an earlier message said the opposite.

CONTRADICTION RULE: If two messages genuinely contradict each other about the same component AND the contradiction has not yet been flagged, return verdict "insufficient" and explain the specific contradiction clearly — quote both statements and ask the clinician to resolve it. Do this ONCE. Do not ask about it again across turns.

"{description}"

═══════════════════════════════════════════════════════════
CRITICAL PRINCIPLE — INFERENCE IS NOT CONFIRMATION:
Every differentiating component must be EXPLICITLY stated. You may NOT infer, imply, or deduce:
  - "reflux in tributaries" does NOT confirm how blood entered the superficial system (C1 MISSING)
  - "blood drains to deep via perforator" does NOT confirm whether the GSV trunk has reflux (C2 status UNKNOWN)
  - describing downstream events does NOT confirm the entry source
  - "perforator" mentioned as an outflow does NOT count as a perforator ENTRY into the GSV
UNKNOWN = INSUFFICIENT. Each required component must be explicitly confirmed YES or NO.
═══════════════════════════════════════════════════════════

STEP 1 — CHECK FOR BASIC VALIDITY FIRST:
Reject immediately (insufficient or question) if:
  - This is a question, greeting, or non-patient input → verdict: "question"
  - Only severity grades ("Reflux grade 2"), diagnosis labels ("GSV incompetence"), or
    symptoms ("leg swelling") with no flow path described → verdict: "insufficient"
  - Only ONE flow transition named with no entry source AND no downstream stated → verdict: "insufficient"

STEP 2 — IDENTIFY WHAT IS EXPLICITLY STATED:
Read the description for its MEANING, not for exact keywords. A clinician will use natural language.
"The GSV carries this blood backward down the thigh" means the same as "reflux in the GSV trunk" — both confirm C2=YES.
"Explicit" means the clinician said it clearly in any natural wording — not that they used a specific phrase from a list.
What is NOT explicit: vague hints, incomplete sentences with no clear direction, or describing one component
and merely implying another (e.g. describing downstream events without stating the entry source).

  COMPONENT 1 — ENTRY SOURCE (how blood enters the superficial system):
    Confirmed SFJ/Hunterian entry:  "SFJ incompetent", "deep blood enters GSV", "blood enters GSV at groin",
                                     "Hunterian perforator incompetent", "deep blood enters GSV at mid-thigh"
    Confirmed perforator entry:     "perforator enters the GSV", "perforator feeds GSV mid-segment",
                                     "perforating vessel inserts into the GSV" (SFJ is competent)
    Confirmed GSV overflow only:    "GSV overflows forward into a tributary", "blood overflows from the GSV into a tributary"
                                     with NO SFJ/perforator entry stated (Type 2A pattern).
                                     *** "reflux in tributaries" alone does NOT confirm this — the GSV overflow must be explicitly stated. ***
    Confirmed no entry/no shunt:    ANY statement that explicitly rules out all pathological entry. This includes:
                                     "no reflux anywhere", "SFJ competent, no perforator entry",
                                     "no retrograde flow detected anywhere", "no retrograde flow anywhere in the limb",
                                     "blood does not enter the superficial system",
                                     "no entry point for blood into the superficial system",
                                     "there is no entry point", "no blood entering the superficial system",
                                     "venous system is competent throughout with no reflux",
                                     "GSV is patent and non-refluxing with no retrograde flow",
                                     "no pathological shunting", "no shunt present",
                                     "no reflux in the GSV or tributaries"
                                     *** If the description confirms NO reflux in GSV trunk AND NO escape into tributaries
                                         AND explicitly states blood does not enter the superficial system
                                         — that IS a confirmed "no entry" case. Do NOT ask how blood enters when the
                                         description has already confirmed it does not enter at all. ***
    NOT CONFIRMED: if none of the above is explicitly stated → COMPONENT 1 MISSING

  COMPONENT 2 — GSV TRUNK REFLUX STATUS (does RP N2→N1 exist?):
    *** SEMANTIC RULE: Any phrase that describes blood moving BACKWARD, DOWNWARD against normal flow,
        or in RETROGRADE direction THROUGH or IN or ALONG the GSV trunk confirms C2 = YES.
        You are reading for MEANING, not matching keywords. ***

    Confirmed YES — any of these meanings expressed in any wording:
      - Blood travels backward / downward / in reverse / in retrograde direction through/along/in the GSV
      - The GSV carries / conducts / transmits blood backward or downward
      - Backward flow / retrograde flow exists in the GSV
      - The GSV is incompetent and blood flows down it
      Examples: "GSV carries this blood backward down the thigh", "blood flows backward along the GSV",
                "GSV conducts blood downward", "retrograde flow in the GSV", "reflux in the GSV trunk",
                "blood moves backward through the GSV", "GSV carries reflux down the leg",
                "blood travels backward through the GSV trunk", "full-length reflux in the GSV"

    Confirmed NO — any of these meanings expressed in any wording:
      - No backward / retrograde flow in the GSV
      - GSV does not carry blood backward
      - GSV trunk is competent / normal flow only
      Examples: "no reflux in the GSV trunk", "GSV does not carry blood backward",
                "no retrograde flow in the GSV", "GSV trunk is competent", "no GSV trunk reflux"

    NOT CONFIRMED: if neither YES nor NO meaning is expressed → COMPONENT 2 MISSING
    REQUIRED WHEN: COMPONENT 1 is SFJ entry, Hunterian entry, or perforator entry.
    NOT REQUIRED WHEN: COMPONENT 1 is "GSV overflow only" (Type 2A) with explicit no-reflux-anywhere.

  COMPONENT 3 — TRIBUTARY ESCAPE STATUS (does EP N2→N3 exist?):
    *** SEMANTIC RULE: Any phrase meaning blood leaves the GSV and enters a side branch / tributary
        confirms C3 = YES. Any phrase meaning there is no branch involvement confirms C3 = NO.
        Read for MEANING, not keywords. ***

    Confirmed YES — blood leaves the GSV into a tributary/branch, in any wording:
      "blood escapes into a tributary", "GSV feeds a branch", "blood exits into a tributary",
      "blood spills into a side branch", "tributary fills from the GSV", "branch fills with blood",
      "blood enters a tributary branch", "GSV discharges into a branch"

    Confirmed NO — no branch involvement, in any wording:
      "no tributary involvement", "blood does not escape into any tributary",
      "blood does not enter any branch", "no branch filling", "tributaries not involved",
      "blood stays within the GSV", "no escape into tributaries", "no blood escaping into the tributary"

    NOT CONFIRMED: if neither YES nor NO meaning is expressed → COMPONENT 3 MISSING
    REQUIRED WHEN: COMPONENT 1 is SFJ entry, Hunterian entry, or perforator entry.
    NOT REQUIRED WHEN: COMPONENT 1 is "GSV overflow only" (already implies tributary escape).

  COMPONENT 4 — TRIBUTARY REFLUX STATUS (does RP N3 exist?):
    *** SEMANTIC RULE: Any phrase meaning blood flows backward / in retrograde direction
        through or in a tributary confirms C4 = YES. ***

    Confirmed YES — backward/retrograde flow in the tributary, in any wording:
      "blood refluxes backward through the tributary", "reflux in the tributary",
      "blood flows backward through the tributary", "tributary carries blood backward",
      "blood drains backward through the branch", "retrograde flow in the tributary"

    Confirmed NO — no backward flow in the tributary, in any wording:
      "no reflux in the tributary", "blood does not flow backward through the tributary",
      "no tributary reflux", "tributary is competent"

    NOT CONFIRMED: if neither YES nor NO meaning is expressed → COMPONENT 4 MISSING
    REQUIRED WHEN: COMPONENT 3 is confirmed YES (tributary escape present).
    NOT REQUIRED WHEN: COMPONENT 3 is confirmed NO.

NOTE ON ELIMINATION TEST: Do NOT require the elimination test as a sufficiency condition.
When all four components above are confirmed (including RP N3 = YES and GSV trunk reflux = YES
alongside SFJ entry and tributary escape), the description is SUFFICIENT — the classification
engine will correctly return UNDETERMINED and request the elimination test itself.

STEP 3 — DETERMINE VERDICT:
A description is SUFFICIENT only when every REQUIRED component from Step 2 is explicitly
confirmed YES or NO. Any REQUIRED component that is NOT CONFIRMED → INSUFFICIENT.

═══════════════════════════════════════════════════════════
WORKED EXAMPLES:

"SFJ incompetent, blood enters GSV, reflux present in GSV trunk full-length"
  C1=SFJ entry ✓  C2=reflux in GSV trunk YES ✓  C3=tributary escape? NOT STATED ✗
  → MISSING: whether blood also escapes into any tributary. → INSUFFICIENT

"SFJ incompetent, reflux in GSV trunk, no tributary involvement confirmed"
  C1=SFJ entry ✓  C2=reflux in GSV trunk YES ✓  C3=NO tributaries ✓  C4=N/A
  → All required components addressed. → SUFFICIENT (Type 1)

"SFJ incompetent, blood escapes into a tributary, blood refluxes backward through the tributary"
  C1=SFJ entry ✓  C2=reflux in GSV trunk? NOT STATED ✗  C3=YES ✓  C4=YES ✓
  → MISSING: whether blood also refluxes backward through the GSV trunk. → INSUFFICIENT

"SFJ incompetent, blood escapes into a tributary, blood refluxes backward through the tributary, no reflux in the GSV trunk"
  C1=SFJ entry ✓  C2=NO ✓  C3=YES ✓  C4=YES ✓
  → All required components addressed. → SUFFICIENT (Type 3)

"SFJ incompetent, reflux in GSV trunk, blood escapes to a tributary"
  C1=SFJ entry ✓  C2=YES ✓  C3=YES ✓  C4=tributary reflux? NOT STATED ✗
  → MISSING: whether blood also refluxes backward through that tributary. → INSUFFICIENT

"SFJ incompetent, reflux in GSV trunk, blood escapes to a tributary, blood also refluxes backward through the tributary"
  C1=SFJ entry ✓  C2=YES ✓  C3=YES ✓  C4=YES ✓
  → All present; elimination test absent but NOT a sufficiency requirement.
  → Classification engine will return UNDETERMINED. → SUFFICIENT

"Perforator enters the GSV at mid-thigh, blood refluxes backward through the tributary"
  C1=perforator entry ✓  C2=reflux in GSV trunk? NOT STATED ✗  C3/C4=tributary reflux ✓
  → MISSING: does blood also reflux backward through the GSV trunk? → INSUFFICIENT

"Perforator enters the GSV at mid-thigh, blood refluxes backward through the tributary, no reflux in the GSV trunk"
  C1=perforator entry ✓  C2=NO ✓  C3/C4=tributary reflux ✓
  → All required components addressed. → SUFFICIENT (Type 2B)

"GSV overflows forward into a tributary at mid-thigh, no reflux detected anywhere"
  C1=GSV overflow ✓  C2=no reflux (implied by no-reflux-anywhere) ✓  C3=YES ✓  C4=NO ✓
  → Sufficient for Type 2A. → SUFFICIENT
═══════════════════════════════════════════════════════════

Return ONE verdict:
"sufficient"   — all required components explicitly confirmed YES or NO
"insufficient" — one or more required components not explicitly addressed
"question"     — not a patient description (greeting, abstract question, etc.)

If "insufficient", the "missing" field MUST be written in plain, natural clinical language — no component numbers, no technical labels, no bullet points, NO pleasantries or acknowledgements.
Write it as two parts in a single flowing response:

PART 1 — CHIVA INTERPRETATION OF WHAT WAS PROVIDED:
Open with "Interpreted so far:" followed by a concise CHIVA summary of every flow event that WAS confirmed. Use plain clinical language with CHIVA notation in parentheses. One sentence per confirmed flow event. Do not mention anything that was NOT confirmed.
Example: "Interpreted so far: Blood enters the GSV at the groin via an incompetent SFJ (EP N1→N2). No blood escapes into any tributary (no EP N2→N3)."

PART 2 — WHAT IS STILL MISSING OR CONTRADICTORY:
On a new line, either:
  (a) State what additional information is still needed and WHY it matters, then give a concrete example of a complete description. OR
  (b) If there is a contradiction between two messages about the same thing, quote BOTH statements explicitly and ask the clinician to resolve it. Do this ONCE — never repeat the same contradiction question.

Do not open Part 2 with "Thanks", "I can see", "Based on what you've provided", or any preamble. Get straight to the point.

GOOD EXAMPLE — missing info:
"Interpreted so far: Blood enters the GSV at the groin via an incompetent SFJ (EP N1→N2).\n\nTo distinguish between Type 1, Type 3, and combined patterns I still need to know: is there reflux in the GSV trunk — does blood travel backward through it? And does any blood escape sideways into a tributary branch? If a tributary is involved, does blood reflux through it as well? For example: 'SFJ incompetent, blood enters the GSV at the groin, reflux present in the GSV trunk full-length, blood escapes into a mid-thigh tributary, and blood refluxes backward through that tributary toward the knee.'"

GOOD EXAMPLE — contradiction:
"Interpreted so far: A perforator at the groin inserts into the GSV (EP N2→N2). Reflux present in the GSV trunk (RP N2→N1).\n\nThere's a contradiction to resolve: in your first message you said 'a tributary at mid-thigh refluxes backward toward the GSV', but later you said 'no blood escapes into the tributary'. If blood is refluxing through the tributary it must have entered it first — can you clarify: is there actually a tributary involved, or was that initial mention a mistake?"

BAD EXAMPLE — repeating the same question multiple turns in a row:
Turn 3: "Does blood reflux through the tributary?"
Turn 4: "Does blood reflux through the tributary?"
Turn 5: "Does blood reflux through the tributary?"
← NEVER do this. Ask once; if answered, accept the answer.

Output ONLY valid JSON — no markdown:
{{"verdict": "sufficient"}}
or
{{"verdict": "insufficient", "missing": "<natural language paragraph as described above>"}}
or
{{"verdict": "question"}}"""

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


def _build_accumulated_description(history: list[dict] | None, current_message: str) -> str:
    """
    Collect user messages from the current classification attempt (since the last
    successful [Analysis] turn) and format them as numbered sequential messages so
    the LLM can resolve contradictions by trusting the most recent statement.
    """
    if not history:
        return current_message

    prior_user_msgs: list[str] = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "assistant" and content.startswith("[Analysis]"):
            prior_user_msgs = []
        elif role == "user":
            prior_user_msgs.append(content.strip())

    all_msgs = prior_user_msgs + [current_message.strip()]
    all_msgs = [m for m in all_msgs if m]

    if len(all_msgs) == 1:
        return all_msgs[0]

    # Number each message so the LLM sees chronological order clearly
    return "\n".join(f"[Message {i+1}]: {m}" for i, m in enumerate(all_msgs))


def parse_nl_to_clips(user_message: str, call_llm_fn: Callable, history: list[dict] | None = None) -> dict:
    """
    Two-stage pipeline:
      1. Focused sufficiency check — a separate call whose only job is to decide
         whether the input describes actual blood movement. No CHIVA clip context,
         so the model can't rationalise sufficiency to justify generating clips.
      2. Full CHIVA interpretation — only reached if stage 1 passes.

    history — prior messages in the session (user + assistant), used to accumulate
    the full clinical description across multiple turns.
    """
    accumulated = _build_accumulated_description(history, user_message)

    # ── Stage 1: sufficiency check ──────────────────────────────────────────
    try:
        check_raw, _ = call_llm_fn(
            _SUFFICIENCY_PROMPT.format(description=accumulated),
            return_usage=True,
            max_tokens=800,
        )
        check = json.loads(_clean_json(check_raw))
        verdict = check.get("verdict", "sufficient")
    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}")
        verdict = "sufficient"  # fall through to CHIVA call on error

    if verdict == "question":
        return {"is_clinical": False, "sufficient_information": False, "missing_information": None, "interpretation": None, "clips": []}

    if verdict == "insufficient":
        missing = check.get("missing") or (
            "Still need to know whether blood refluxes backward through the GSV trunk, "
            "whether it escapes into any tributary, and if so whether it also refluxes "
            "backward through that tributary."
        )
        return {"is_clinical": True, "sufficient_information": False, "missing_information": missing, "interpretation": None, "clips": []}

    # ── Stage 2: CHIVA interpretation (only if verdict == "sufficient") ─────
    try:
        raw, _ = call_llm_fn(
            _NL_TO_CHIVA_PROMPT.format(description=accumulated),
            return_usage=True,
            max_tokens=1024,
        )
        result = json.loads(_clean_json(raw))
        if isinstance(result, dict) and "clips" in result:
            if _has_no_reflux_statement(accumulated) and result.get("clips"):
                before = len(result["clips"])
                result["clips"] = [c for c in result["clips"] if c.get("flow") != "RP"]
                after = len(result["clips"])
                if before != after:
                    logger.info(f"Stripped {before - after} hallucinated RP clip(s) — explicit no-reflux statement.")
            return {
                "is_clinical": True,
                "sufficient_information": True,
                "missing_information": None,
                "interpretation": result.get("interpretation"),
                "clips": result.get("clips", []),
            }
    except Exception as e:
        logger.error(f"CHIVA interpretation failed: {e}")
    return {"is_clinical": False, "sufficient_information": False, "missing_information": None, "interpretation": None, "clips": []}


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
