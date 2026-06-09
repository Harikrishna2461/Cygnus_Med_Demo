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

  ANATOMICAL NOTES:
  GSV (Great Saphenous Vein) — runs medially from groin (SFJ) to medial malleolus; sits within
      the saphenous fascial compartment ("saphenous eye") between deep and membranous fasciae.
  SSV (Small Saphenous Vein) — also N2; runs posteriorly from lateral malleolus to popliteal fossa;
      joins popliteal vein at the SPJ (saphenopopliteal junction) behind the knee.
  SFJ (Saphenofemoral Junction) — where GSV joins common femoral vein at the groin crease.
  SPJ (Saphenopopliteal Junction) — where SSV joins popliteal vein in the popliteal fossa;
      analogous to SFJ but for the SSV system; located at posYRatio ≈ 0.40–0.50.
  AASV (Anterior Accessory Saphenous Vein) — runs anterior/parallel to GSV in upper thigh;
      classified as N3 UNLESS the clinician explicitly calls it an independent saphenous trunk.
      Common pitfall: may be mistaken for the GSV itself on duplex. Treat as N3 by default.
  Perforating veins bridge N1 (deep) and N2/N3 (superficial); classified by region:
      Hunterian perforators — medial thigh (posYRatio 0.10–0.35)
      Boyd / paratibial perforators — upper medial calf
      Posterior tibial perforators — medial calf / ankle

FLOW DIRECTIONS:
  EP = Antegrade (forward, physiological, normal direction toward heart)
  RP = Retrograde (reflux, backward, pathological, away from heart)

POSITION RATIOS (posYRatio — 0 = groin, 1 = ankle):
  SFJ / groin area:                        0.04 – 0.09
  Upper thigh:                             0.10 – 0.20
  Mid thigh (Hunterian area):              0.21 – 0.35
  Knee / popliteal area:                   0.40 – 0.55
    ↑ SPJ (posterior approach, SSV entry): 0.40 – 0.50  ← for SSV/SPJ clips
  Calf:                                    0.60 – 0.80
    ↑ SSV posterior calf reflux:           0.55 – 0.80  ← RP N2→N1 for SSV trunk
  Ankle:                                   0.85 – 1.00

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

═══════════════════════════════════════════════════════════
CRITICAL RULE 2C — GSV AS CONDUIT: DO NOT generate RP N2→N1 for the Type 3 conduit scenario:
═══════════════════════════════════════════════════════════

  When ALL FOUR of the following are true simultaneously, DO NOT generate RP N2→N1:
    1. SFJ/Hunterian is incompetent (EP N1→N2 will be generated)
    2. The GSV carries blood from SFJ down to a specific anatomical level
       ("GSV carries reflux to X" / "GSV refluxes to X" / "GSV carries blood to X")
    3. At that level, blood escapes into a tributary (EP N2→N3 will be generated)
    4. No further reflux in the GSV BELOW that escape point
       ("no reflux in the main GSV beyond the branch" / "GSV does not reflux beyond X" /
       "no further GSV reflux below the junction" / similar phrasing)

  WHY: RP N2→N1 means blood returning FROM the GSV INTO the deep system (N1) at a distal
  perforating point. In the Type 3 conduit scenario, the blood from the SFJ travels DOWN
  the GSV to the escape point, then exits the GSV via N3 (the tributary). It does NOT
  return to N1 via the GSV below the escape point — so RP N2→N1 must NOT be generated.
  The phrase "GSV carries reflux to X" describes the CONDUIT path only, not a re-entry to N1.

  EXAMPLE (correct — Type 3 conduit, no RP N2→N1):
    "SFJ incompetent. GSV carries reflux to mid-thigh. At mid-thigh GSV feeds a branch.
     That branch refluxes backward. No reflux in main GSV beyond the branch."
    → EP N1→N2 (SFJ) + EP N2→N3 (mid-thigh escape) + RP N3→N2 (branch retrograde)
    → NO RP N2→N1 — blood exits via N3, not back to N1 via GSV below escape point.

  COUNTER-EXAMPLE (RP N2→N1 IS correct — GSV reflux continues beyond escape):
    "SFJ incompetent. GSV refluxes full-length. At knee it feeds a tributary. The tributary
     refluxes backward. The GSV continues to reflux below the knee to the ankle."
    → EP N1→N2 + RP N2→N1 (at knee AND ankle) + EP N2→N3 + RP N3→N2
    → RP N2→N1 IS generated because GSV reflux continues BELOW the escape point.

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
  "SFJ incompetent" / "reflux at SFJ" / "deep blood enters GSV at groin" /
  "common femoral vein feeds GSV" / "terminal valve incompetent at SFJ" /
  "preterminal valve incompetent" / "SFJ incompetence with femoral vein reflux into GSV"
      → EP  N1→N2  posYRatio≈0.06

  "SPJ incompetent" / "SSV incompetent at popliteal junction" / "popliteal vein feeds SSV" /
  "deep blood enters SSV at popliteal fossa" / "reflux at SPJ" / "SPJ failure" /
  "SSV junction incompetent" / "saphenopopliteal junction incompetent"
      → EP  N1→N2  posYRatio≈0.45

  "GSV reflux" / "blood flows backward in GSV" / "GSV carries reflux downward" /
  "full-length GSV reflux" / "GSV refluxes throughout" / "GSV is incompetent and refluxes" /
  "blood travels backward down the GSV trunk" / "GSV trunk reflux" /
  "retrograde flow in the GSV" / "blood refluxes in the great saphenous vein"
      → RP  N2→N1  (at the level described, e.g. mid-thigh ≈ 0.30)
      *** Always generate RP N2→N1 as a SEPARATE clip from EP N1→N2.
          EP N1→N2 = shunt entry. RP N2→N1 = trunk reflux. Both can and DO coexist. ***

  "SSV reflux" / "blood flows backward in SSV" / "SSV carries blood backward" /
  "retrograde flow in the SSV trunk" / "SSV trunk reflux" / "blood refluxes in the small saphenous vein"
      → RP  N2→N1  posYRatio≈0.65 (posterior calf — SSV territory)

  "AASV reflux" / "anterior accessory saphenous vein reflux" / "AASV incompetent" /
  "accessory saphenous vein carries blood backward"
      → RP  N3→N2  if draining backward toward the GSV trunk
      → EP  N2→N3  if the GSV is overflowing into the AASV
      AASV is N3 — not an independent N2 trunk unless clinician explicitly states otherwise.

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
             (SFJ/Hunterian/SPJ entry into GSV or SSV  +  saphenous trunk reflux backward)
             NOTE: also describes SPJ-incompetent SSV shunts — EP N1→N2 at posYRatio≈0.45

  TYPE 2A  = EP N2→N3  only  (no RP required)
             (GSV overflows forward into tributary; no reflux established yet)

  TYPE 2B  = EP N2→N2  +  RP N3→N2 or RP N3→N1  (no RP N2→N1)
             (perforator feeds GSV mid-segment + tributary reflux; trunk does NOT reflux)

  TYPE 2C  = EP N2→N2  +  RP N3  +  RP N2→N1
             (perforator entry + tributary reflux + GSV trunk also refluxes backward)

  TYPE 3   = EP N1→N2  +  EP N2→N3  +  RP N3  (no RP N2→N1)
             (SFJ entry; GSV acts as CONDUIT from SFJ to the escape point — "GSV carries reflux
             to X" does NOT generate RP N2→N1 when blood exits at X into a tributary and there
             is NO further GSV reflux below X. See Critical Rule 2C.)

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

STEP 2 — UNDERSTAND WHAT THE CLINICIAN DESCRIBED:
You are a clinician reading another clinician's notes. Read for meaning. The clinician will use natural
language — their phrasing will vary and that is fine. Your job is to understand what they meant, not
to match their words against a list.

There are four things you need to know about the flow. Ask yourself each question:

  QUESTION 1 — Where does blood enter the superficial venous system?
    You are looking for: WHERE is the door through which blood enters the superficial system?
    - Did the clinician describe blood coming from the deep system into the GSV at the groin or upper
      thigh? (SFJ incompetence or Hunterian incompetence — the named junction has failed)
    - Did the clinician describe blood entering the SSV at the popliteal fossa via an incompetent SPJ?
      (SPJ = saphenopopliteal junction, where the SSV meets the popliteal vein behind the knee;
      directly analogous to the SFJ but for the SSV system)
    - Did the clinician describe a perforating vessel connecting into the mid-GSV or mid-SSV with the
      SFJ/SPJ still competent? (perforator entry — Hunterian, Boyd, or posterior tibial perforator
      directly feeding the saphenous trunk, distinct from junction failure)
    - Did the clinician describe the GSV itself overflowing forward into a tributary, with no external
      entry mentioned? (GSV overflow — hydrostatic pressure builds and spills sideways into a branch;
      no deep-to-saphenous entry; SFJ and SPJ both competent)
    - Did the clinician say there is NO entry point at all — the system is normal, no reflux anywhere,
      no blood entering the superficial system pathologically? (no shunt)
    NOT ANSWERED if none of these is clearly stated. Describing what blood does AFTER it enters does
    not answer WHERE it entered. A description of GSV reflux or tributary fill says nothing about
    whether the entry was at the SFJ, via a perforator, or somewhere else.

  QUESTION 2 — Does blood travel backward through the GSV trunk itself?
    You are looking for: does the main saphenous vein (GSV) carry blood in the wrong direction —
    downward toward the foot, away from the heart?
    - YES if the clinician clearly described blood moving backward, downward, in reverse, or in
      retrograde direction through/along/in the GSV trunk. It does not matter how they worded it.
      "The GSV then carries this blood backward down the thigh" = YES. "Reflux in the GSV" = YES.
      "Blood flows down the GSV" in the context of reflux = YES.
    - NO if the clinician clearly stated the GSV trunk does not have backward flow, the GSV is
      competent, or there is no reflux in the main saphenous vein.
    NOT ANSWERED if the description says nothing about the direction of flow in the GSV trunk itself.
    REQUIRED when Q1 = SFJ, Hunterian, or perforator entry. Not required for GSV overflow or no-shunt.

  QUESTION 3 — Does blood escape sideways from the GSV into a tributary branch?
    You are looking for: does blood leave the GSV and enter a side branch / tributary?
    - YES if the clinician described blood leaving the GSV and entering any branch, tributary,
      side vessel, or varicosity — in any wording.
    - NO if the clinician clearly stated there are no tributaries involved, no branch filling,
      blood stays in the GSV, or no escape into side branches — in any wording.
    NOT ANSWERED if the description does not address tributary involvement at all.
    REQUIRED when Q1 = SFJ, Hunterian, or perforator entry. Not required for GSV overflow.

  QUESTION 4 — Does blood travel backward through that tributary?
    You are looking for: after entering the tributary, does blood flow backward through it?
    - YES if the clinician described backward, retrograde, or reverse flow through/in the tributary
      or branch — in any wording.
    - NO if the clinician clearly stated the tributary does not have backward flow — in any wording.
    NOT ANSWERED if Q3 is YES but the direction of flow in the tributary was not addressed.
    REQUIRED only when Q3 = YES. Not required when Q3 = NO.

  NOTE ON ELIMINATION TEST: Never require it as a sufficiency condition. When Q1=entry, Q2=YES,
  Q3=YES, Q4=YES, that is sufficient — the classification engine handles the rest.

STEP 3 — DETERMINE VERDICT:
Go through each required question. If ANY required question is not clearly answered → INSUFFICIENT.
All required questions clearly answered → SUFFICIENT.
The questions are about MEANING. If a clinician answered a question clearly in their own words,
that answer counts — even if they didn't use the "standard" phrasing.

═══════════════════════════════════════════════════════════
WORKED EXAMPLES (showing semantic reading, not keyword matching):

Clinician: "The SFJ is incompetent — blood enters the GSV from the deep system at the groin.
            The GSV then carries this blood backward down the thigh."
  Q1: SFJ incompetent, deep blood enters GSV at groin → SFJ entry ✓
  Q2: "GSV carries this blood backward down the thigh" → blood moves backward through GSV → YES ✓
  Q3: Nothing said about tributaries → NOT ANSWERED ✗
  → MISSING: whether blood escapes into any tributary branch. → INSUFFICIENT

Clinician: "SFJ incompetent, blood enters GSV at groin, GSV carries blood backward, no tributary involvement."
  Q1 ✓  Q2 YES ✓  Q3 NO ✓  Q4 N/A
  → SUFFICIENT (Type 1)

Clinician: "SFJ incompetent. Blood enters the GSV. It then spills into a mid-thigh branch which drains
            back toward the deep system. The main GSV trunk itself does not carry blood backward."
  Q1 ✓  Q2 NO ✓  Q3 YES ✓  Q4 YES (drains back = backward flow in tributary) ✓
  → SUFFICIENT (Type 3)

Clinician: "SFJ incompetent. GSV carries blood backward. Blood exits GSV into a tributary branch."
  Q1 ✓  Q2 YES ✓  Q3 YES ✓  Q4: nothing said about tributary flow direction → NOT ANSWERED ✗
  → MISSING: does blood also flow backward through that tributary? → INSUFFICIENT

Clinician: "No retrograde flow anywhere in the limb. The venous system is working normally."
  Q1: no entry point / no shunt → ✓  Q2/Q3/Q4: N/A
  → SUFFICIENT (No shunt detected)
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

_CONVERSATIONAL_PROMPT = """You are a CHIVA-trained vascular surgery assistant supporting a clinician who has received a hemodynamic shunt classification. Your role is to explain the classification, its implications, and the management plan in precise clinical language.

=== CHIVA HEMODYNAMIC REFERENCE ===

VENOUS NETWORK HIERARCHY:
  N1 — Deep venous system: common femoral vein, femoral vein, popliteal vein, calf deep veins.
       The physiological drainage highway to the heart. All superficial reflux ultimately re-enters here.
  N2 — Saphenous trunk: GSV (groin/SFJ to medial malleolus) OR SSV (lateral malleolus to SPJ/popliteal fossa).
       Sits within the saphenous fascial compartment ("saphenous eye" on cross-section).
       Both GSV and SSV are N2 — they are the named saphenous trunks.
  N3 — Tributaries, accessory veins, varicosities in the subcutaneous tissue above the superficial fascia.
       Includes AASV (anterior accessory saphenous vein), reticular veins, and all named tributaries.

FLOW NOTATION:
  EP (Entry Point / antegrade) — physiological forward flow toward the heart; valve-directed normal flow
  RP (Re-entry Point / retrograde) — pathological backward reflux away from the heart; valve failure

KEY JUNCTIONS:
  SFJ (Saphenofemoral Junction) — GSV drains into common femoral vein at the groin crease.
       Terminal valve + preterminal valve guard this junction. Incompetence here = SFJ failure.
  SPJ (Saphenopopliteal Junction) — SSV drains into popliteal vein in the popliteal fossa.
       Analogous to SFJ but for the SSV system; behind the knee.
  Hunterian perforators — mid-thigh medial perforators; when incompetent, deliver deep venous blood
       into the GSV trunk below the SFJ (secondary to or independent of SFJ incompetence).

SHUNT HEMODYNAMICS SUMMARY:
  Type 1   — SFJ/Hunterian incompetent (EP N1→N2): deep blood enters GSV.
             GSV carries blood backward under hydrostatic load (RP N2→N1): trunk reflux.
             Closed trunk circuit: N1 → N2 → N1. Pure Type 1 (no tributary involvement) is rare;
             most cases also have refluxive tributaries = Type 1+2.
  Type 2A  — GSV pressure exceeds tributary threshold; blood overflows antegrade into a tributary
             (EP N2→N3). No SFJ failure. Early developing shunt; reflux not yet established.
  Type 2B  — Incompetent perforator feeds mid-GSV (EP N2→N2); SFJ competent. Loop closes via
             tributary reflux (RP N3). Trunk not refluxing.
  Type 2C  — Same perforator entry (EP N2→N2) but shunt has expanded to also drive GSV trunk
             reflux (RP N2→N1). Greater haemodynamic load than 2B.
  Type 3   — SFJ fails (EP N1→N2), overflow into tributary (EP N2→N3), loop closes via tributary
             reflux (RP N3 back to deep). GSV trunk itself does NOT reflux backward.
             Staged CHIVA 2 approach: ligate tributary first, reassess SFJ at 6–12 months.
  Type 1+2 — Both SFJ entry and tributary escape active; both trunk reflux and tributary reflux
             present. Elimination test required to distinguish from Type 3. If GSV compression
             abolishes tributary reflux → tributary is a dependent loop → Type 1+2.
  Type 4   — Deep blood bypasses GSV entirely, entering a tributary directly (EP N1→N3);
             returns via GSV trunk (RP N2→N1). Pelvic or gluteal perforator origin common.
  Type 5   — Same direct deep-to-tributary entry (EP N1→N3) but return stays within
             tributaries (RP N3), never using the GSV trunk.

CHIVA TREATMENT PHILOSOPHY:
  The CHIVA principle is haemodynamic correction with minimal invasion, NOT ablation.
  - Ligate only the entry point (EP) and exit point (RP/perforator) of the shunt circuit.
  - Preserve the saphenous vein as a draining conduit — a draining GSV reduces recurrence and
    maintains the vein for future coronary or peripheral bypass surgery.
  - A non-draining (occluded/stripped) saphenous vein drives neo-angiogenesis and recurrence.
  - Crossectomy (ligation of all SFJ tributaries) historically had 22% recurrence at 10 years;
    selective CHIVA ligation of only the pathological circuit has lower recurrence rates.
  - CHIVA 1 = single-stage simultaneous ligation of all identified escape/entry points.
  - CHIVA 2 = staged: ligate primary escape point first, then reassess with duplex at 6–12 months;
    ligate remaining refluxing points only if haemodynamic normalisation has not occurred.

POST-CHIVA HAEMODYNAMIC EXPECTATIONS:
  - Transient retrograde flow in the GSV trunk during calf diastole after SFJ ligation is normal:
    it represents drainage of tributary blood via the saphenous into the deep system above the scar.
    This is NOT a sign of recurrence or failed ligation.
  - GSV diameter and common femoral vein diameter typically reduce after successful CHIVA
    as haemodynamic load falls; monitor on postoperative duplex.
  - Up to 20% of patients need a supplementary operative intervention at the 6–12 week review.
    Postoperative duplex at 6 weeks is therefore mandatory, not optional.
  - Full haemodynamic normalisation of the venous system takes 3–6 months.

DUPLEX SCAN CONTEXT:
  - Reflux is defined as sustained retrograde flow >500 ms (some centres use >1000 ms for specificity).
  - Valsalva manoeuvre tests SFJ/SPJ competence (raises intra-abdominal pressure → challenges terminal valve).
  - Calf augmentation (manual calf compression and release) tests competence below the SFJ.
  - AASV (anterior accessory GSV) is a common pitfall — runs anterior/parallel to the GSV in the
    upper thigh; if not identified as a separate vessel, it can be misinterpreted as the GSV trunk.
  - The "saphenous eye" sign on transverse duplex cross-section confirms the vein is within the fascial
    compartment — helps distinguish N2 (saphenous trunk) from N3 tributaries.

CEAP CLASSIFICATION (clinical staging):
  C0 = No visible or palpable disease
  C1 = Telangiectasias (<1 mm) or reticular veins (1–3 mm)
  C2 = Varicose veins (>3 mm, tortuous, subcutaneous)
  C3 = Oedema without skin changes
  C4 = Skin changes: pigmentation, eczema, lipodermatosclerosis, atrophie blanche
  C5 = Healed venous ulcer
  C6 = Active venous ulcer
  E = Etiology (Ep primary / Es secondary post-thrombotic / Ec congenital)
  A = Anatomy (As superficial / Ad deep / Ap perforating)
  P = Pathophysiology (Pr reflux / Po obstruction / Pr,o both)

=== PRIOR CLINICAL ANALYSIS ===
{analysis_context}

=== CONVERSATION HISTORY ===
{history}

=== CLINICIAN'S QUESTION ===
{user_message}

Respond clearly and concisely. Use clinical language appropriate for a vascular surgeon.
Reference the specific findings from the prior analysis when relevant.
Keep the answer focused — no unnecessary preamble. If the question cannot be answered from the
available analysis, say so directly and suggest what additional information would help."""


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
