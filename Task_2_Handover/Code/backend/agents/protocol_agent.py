"""
Examination Protocol Agent.

Returns the zone-specific duplex ultrasound examination protocol for the
current probe position, sourced from medical literature.

Sources (all verified from PDF reading, June 2026):
  Adler et al. 2022  â€” RadioGraphics: varicose veins evaluation protocols
  Gianesini et al. 2014 â€” Phlebology: CHIVA strategy
  Delfrate 2023 â€” JTAVR: CHIVA duplex assessment protocol
  AVF 2023 guidelines â€” perforator criteria
  Mendoza et al. 2014 â€” Duplex Ultrasound of Superficial Leg Veins
    Ch. 7.2  â€” GSV examination objectives
    Ch. 8.2  â€” SSV examination objectives
    Ch. 9.2  â€” Perforating vein examination objectives
    Ch. 10.2 â€” Tributary examination objectives
    Ch. 14   â€” Deep leg vein (DVT/compression) assessment
"""
from __future__ import annotations


_PROTOCOLS: dict[str, str] = {

    "sfj_groin": (
        "EXAMINATION PROTOCOL â€” SFJ/Groin (Adler 2022 + Gianesini 2014 + Delfrate 2023)\n"
        "1. Patient position: Reverse Trendelenburg â‰¥60Â° to maximise venous filling (Adler 2022).\n"
        "2. Transverse B-mode: 'Mickey Mouse' sign â€” CFV in centre, GSV and femoral artery as lateral ovals.\n"
        "3. Place Doppler sample gate on FEMORAL SIDE of the terminal valve (Gianesini 2014).\n"
        "4. Valsalva maneuver: confirmed adequate when forward CFV flow ceases. Look for flow reversal into GSV.\n"
        "5. Then apply ParanÃ  maneuver (waist push triggers calf proprioceptive reflex â€” more physiological than squeezing).\n"
        "6. BOTH Valsalva AND ParanÃ  must be positive to confirm SFJ incompetence (EP N1â†’N2) (Gianesini 2014).\n"
        "7. If Valsalva NEGATIVE but ParanÃ  positive: terminal valve is competent â€” reflux is pre-terminal or from a pelvic leak point. Check SGP, IGP, OP (Delfrate 2023 p.22).\n"
        "8. If BOTH Valsalva AND ParanÃ  positive: SFJ incompetence confirmed (EP N1â†’N2 at groin) â€” proceed distally along GSV trunk to establish extent of reflux.\n"
        "9. Assess AASV (anterior accessory saphenous vein) separately â€” lies anterior to GSV, classified N3 not N2."
    ),

    "upper_thigh": (
        "EXAMINATION PROTOCOL â€” Upper Thigh / GSV Proximal (Adler 2022)\n"
        "1. Transverse B-mode: confirm GSV sits within 'saphenous eye' (fascial compartment) â€” N2 identity.\n"
        "2. Measure GSV anteroposterior diameter (document at this level).\n"
        "3. Apply ParanÃ /squeeze and release: reflux >500 ms = trunk reflux (RP N2â†’N1).\n"
        "4. AASV may run parallel to GSV in upper thigh â€” assess it separately (N3, not N2).\n"
        "5. After SFJ entry (EP N1â†’N2) is confirmed, the next findings to establish are trunk reflux (RP N2â†’N1) and any trunk-to-tributary escape (EP N2â†’N3) â€” both assessed along the medial thigh."
    ),

    "hunterian_proximal": (
        "EXAMINATION PROTOCOL â€” Proximal Thigh / Hunterian Zone (posY 0.21â€“0.33) (Adler 2022 + Delfrate 2023)\n"
        "Source: Hunterian perforators = proximal/middle thigh, within Hunter's canal (DuplexUS 2014 p.33)\n"
        "1. Transverse B-mode at medial proximal thigh: confirm GSV in fascial compartment ('saphenous eye').\n"
        "2. KEY ZONE FOR EP N1â†’N2: Hunterian perforators connect femoral vein (FV) to GSV within Hunter's canal.\n"
        "   If SFJ is competent but thigh GSV shows reflux â†’ Hunterian perforator is the likely EP N1â†’N2.\n"
        "3. Perforator maneuvers â€” all three required (Delfrate 2023):\n"
        "   a. Static squeezing (gravitational test)\n"
        "   b. ParanÃ  maneuver (physiological â€” preferred over squeezing alone)\n"
        "   c. Valsalva (hypertensive test â€” outward flow = pathological/pathogenic perforator)\n"
        "4. Pathological perforator: outward flow â‰¥500 ms AND diameter â‰¥3.5 mm (AVF 2023).\n"
        "5. Watch for N3 above fascia at same level as N2 in compartment â€” junction is EP N2â†’N3 (trunk escape).\n"
        "6. Trunk reflux visible here without SFJ entry confirms Hunterian perforator as entry point (EP N1â†’N2)."
    ),

    "dodd_distal": (
        "EXAMINATION PROTOCOL â€” Distal Thigh / Dodd Zone (posY 0.34â€“0.47) (Adler 2022 + Delfrate 2023)\n"
        "Source: Dodd perforators = distal third of thigh, just above the knee (DuplexUS 2014 p.33)\n"
        "1. Transverse B-mode at medial distal thigh: confirm GSV in fascial compartment ('saphenous eye').\n"
        "2. Dodd perforators connect the femoral vein (FV) to the GSV just above the knee.\n"
        "3. Perforator maneuvers â€” all three required (Delfrate 2023):\n"
        "   a. Static squeezing (gravitational test)\n"
        "   b. ParanÃ  maneuver (physiological â€” preferred over squeezing alone)\n"
        "   c. Valsalva (hypertensive test â€” outward flow = pathological/pathogenic perforator)\n"
        "4. Pathological perforator: outward flow â‰¥500 ms AND diameter â‰¥3.5 mm (AVF 2023).\n"
        "5. Watch for N3 above fascia at same level as N2 in compartment â€” junction is EP N2â†’N3 (trunk escape).\n"
        "6. Principle: 'No reflux no re-entry' â€” if GSV reflux persists below escape, another re-entry exists distally."
    ),

    "popliteal_spj": (
        "EXAMINATION PROTOCOL â€” Popliteal / SPJ (Gianesini 2014 + Delfrate 2023)\n"
        "1. Position: lateral decubitus (left decubitus for right SSV; right decubitus for left SSV).\n"
        "2. BOTH ParanÃ  (active) AND compression/relaxation (passive CR) must be positive simultaneously\n"
        "   to confirm SPJ incompetence (EP N1â†’N2 at SPJ). One positive alone â‰  incompetence.\n"
        "3. SPJ location is variable â€” may connect to gastrocnemian vein rather than popliteal vein directly (Delfrate 2023).\n"
        "4. Assess Giacomini vein separately (posterior thigh, SSVâ†’GSV connection).\n"
        "   Forward flow in Giacomini during ParanÃ  = viable outflow route.\n"
        "5. When planning surgery: SPJ disconnection should be performed below the Giacomini junction in mixed shunts."
    ),

    "calf": (
        "EXAMINATION PROTOCOL â€” Calf (Adler 2022 + Delfrate 2023)\n"
        "1. Track N3 tributaries distally toward re-entry perforators along medial and posterior surfaces.\n"
        "2. ParanÃ  maneuver: inward perforator flow during muscle DIASTOLE (relaxation) = re-entry point (RP N3â†’N1).\n"
        "   Diastolic reflux into deep system via perforator is always pathological and pathogenic.\n"
        "3. Biphasic perforator: systolic outward flow followed by diastolic inward flow = likely re-entry candidate.\n"
        "   The diastolic inflow is the haemodynamically significant phase (Delfrate 2023).\n"
        "4. Pathological perforator (AVF 2023): outward flow â‰¥500 ms AND diameter â‰¥3.5 mm.\n"
        "5. Squeezing alone is insufficient â€” use ParanÃ  (proprioceptive, physiological) as primary maneuver.\n"
        "6. Medial calf perforators (paratibial, posterior tibial) are the most common GSV re-entry sites (Mendoza 2014 Ch. 9.2; Delfrate 2023)."
    ),

    "ankle_ssv": (
        "EXAMINATION PROTOCOL â€” Ankle / Lower Calf (Adler 2022 + Delfrate 2023)\n"
        "1. GSV at medial malleolus (posY 0.85â€“1.00): medial surface, N2 in fascial compartment.\n"
        "2. SSV at lateral ankle: assess in lateral decubitus. Confirm it joins SPJ posteriorly.\n"
        "3. ParanÃ  squeeze/release at distal perforators: inward flow on release = RP N3â†’N1 (circuit closure).\n"
        "4. Distal calf SSV assessment is mandatory when stasis ulcers are present (Adler 2022).\n"
        "5. Confirm diameter and outward flow duration to classify perforators as pathological."
    ),

    "general_sequence": (
        "GENERAL EXAMINATION SEQUENCE (Adler 2022 + Delfrate 2023)\n"
        "Step 1 â€” DVT: compression assessment of all deep veins before any reflux testing.\n"
        "Step 2 â€” Deep reflux: Valsalva for iliac valve competence; CFV reflux check above SFJ.\n"
        "Step 3 â€” SFJ: Mickey Mouse sign; Valsalva + ParanÃ  (both must be positive); AASV separately.\n"
        "Step 4 â€” GSV trunk: medial thigh â†’ calf; saphenous eye in transverse; 500 ms reflux threshold.\n"
        "Step 5 â€” Perforators: Hunterian zone and calf; all 3 maneuvers; note biphasic flow.\n"
        "Step 6 â€” SPJ: posterior knee; both ParanÃ  + CR; variable anatomy (check Giacomini).\n"
        "Step 7 â€” SSV trunk: posterior calf; lateral approach; same 500 ms threshold.\n"
        "Step 8 â€” Re-entry perforators: identify by diastolic inward flow; confirm â‰¥3.5 mm diameter.\n"
        "Patient positioning: Reverse Trendelenburg â‰¥60Â° for ALL reflux studies (Adler 2022)."
    ),
}


def get_protocol(region: str, pos_y: float) -> str:
    """
    Return the examination protocol for the current probe position.

    posY takes priority over region name for intra-region zone selection
    (e.g. GSV-THI at posY 0.28 returns Hunterian protocol, not upper-thigh).

    Args:
        region: Anatomical region string (e.g. "SFJ", "GSV-THI", "GSV-CAL", "SPJ", "SSV").
        pos_y:  Probe posY ratio (0.0 = groin, 1.0 = ankle).

    Returns:
        Multi-line protocol string ready to embed in the LLM state message.
    """
    r = region.upper().replace("_", "-")

    # Named junction regions take precedence regardless of posY.
    if r == "SFJ":
        return _PROTOCOLS["sfj_groin"]
    elif r == "SPJ":
        return _PROTOCOLS["popliteal_spj"]
    elif r == "SSV":
        return _PROTOCOLS["calf"]
    # For all other regions (GSV-THI, GSV-CAL, UNKNOWN, etc.) use posY bands.
    # Boundaries from DuplexUS 2014 p.33-34, Adler 2022.
    elif pos_y <= 0.07:
        return _PROTOCOLS["sfj_groin"]
    elif pos_y <= 0.20:
        return _PROTOCOLS["upper_thigh"]
    elif pos_y <= 0.33:
        return _PROTOCOLS["hunterian_proximal"]
    elif pos_y <= 0.47:
        return _PROTOCOLS["dodd_distal"]
    elif pos_y <= 0.57:
        return _PROTOCOLS["popliteal_spj"]
    elif pos_y <= 0.88:
        return _PROTOCOLS["calf"]
    else:
        return _PROTOCOLS["ankle_ssv"]


