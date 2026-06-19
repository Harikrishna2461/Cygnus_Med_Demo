"""
Examination Protocol Agent.

Returns the zone-specific duplex ultrasound examination protocol for the
current probe position, sourced from medical literature.

Sources (all verified from PDF reading, June 2026):
  Adler et al. 2022  — RadioGraphics: varicose veins evaluation protocols
  Gianesini et al. 2014 — Phlebology: CHIVA strategy
  Delfrate 2023 — JTAVR: CHIVA duplex assessment protocol
  AVF 2023 guidelines — perforator criteria
"""
from __future__ import annotations


_PROTOCOLS: dict[str, str] = {

    "sfj_groin": (
        "EXAMINATION PROTOCOL — SFJ/Groin (Adler 2022 + Gianesini 2014 + Delfrate 2023)\n"
        "1. Patient position: Reverse Trendelenburg ≥60° to maximise venous filling (Adler 2022).\n"
        "2. Transverse B-mode: 'Mickey Mouse' sign — CFV in centre, GSV and femoral artery as lateral ovals.\n"
        "3. Place Doppler sample gate on FEMORAL SIDE of the terminal valve (Gianesini 2014).\n"
        "4. Valsalva maneuver: confirmed adequate when forward CFV flow ceases. Look for flow reversal into GSV.\n"
        "5. Then apply Paranà maneuver (waist push triggers calf proprioceptive reflex — more physiological than squeezing).\n"
        "6. BOTH Valsalva AND Paranà must be positive to confirm SFJ incompetence (EP N1→N2) (Gianesini 2014).\n"
        "7. If Valsalva negative: Paranà reflux indicates pre-terminal valve incompetence or pelvic leak point — NOT SFJ entry.\n"
        "8. If Valsalva positive: follow arch tributaries upward to check for pelvic leak points (SGP, IGP, OP) (Delfrate 2023).\n"
        "9. Assess AASV (anterior accessory saphenous vein) separately — lies anterior to GSV, classified N3 not N2."
    ),

    "upper_thigh": (
        "EXAMINATION PROTOCOL — Upper Thigh / GSV Proximal (Adler 2022)\n"
        "1. Transverse B-mode: confirm GSV sits within 'saphenous eye' (fascial compartment) — N2 identity.\n"
        "2. Measure GSV anteroposterior diameter (document at this level).\n"
        "3. Apply Paranà/squeeze and release: reflux >500 ms = trunk reflux (RP N2→N1).\n"
        "4. AASV may run parallel to GSV in upper thigh — assess it separately (N3, not N2).\n"
        "5. If SFJ entry (EP N1→N2) confirmed, trace GSV distally for trunk reflux (RP N2→N1) or escape (EP N2→N3)."
    ),

    "mid_thigh_dodd": (
        "EXAMINATION PROTOCOL — Mid-Thigh / Dodd Zone (posY 0.21–0.33) (Adler 2022 + Delfrate 2023)\n"
        "Source: Dodd perforators = middle third of thigh (DuplexUS 2014 p.33; Lee 2017 p.129)\n"
        "1. Transverse B-mode at medial thigh: confirm GSV in fascial compartment ('saphenous eye').\n"
        "2. Dodd perforators connect the femoral vein (FV) to the GSV at this level.\n"
        "3. Perforator maneuvers — all three required (Delfrate 2023):\n"
        "   a. Static squeezing (gravitational test)\n"
        "   b. Paranà maneuver (physiological — preferred over squeezing alone)\n"
        "   c. Valsalva (hypertensive test — outward flow = pathological/pathogenic perforator)\n"
        "4. Pathological perforator: outward flow ≥500 ms AND diameter ≥3.5 mm (AVF 2023).\n"
        "5. Watch for N3 above fascia at same level as N2 in compartment — junction is EP N2→N3 (trunk escape).\n"
        "6. If SFJ competent but trunk shows reflux here → move distally to Hunterian zone (posY 0.34–0.47)."
    ),

    "lower_thigh_hunterian": (
        "EXAMINATION PROTOCOL — Lower Thigh / Hunterian Zone (posY 0.34–0.47) (Adler 2022 + Delfrate 2023)\n"
        "Source: Hunterian perforators = DISTAL third of thigh (DuplexUS 2014 p.33-34; Lee 2017 p.129)\n"
        "1. Transverse B-mode at medial distal thigh: confirm GSV in fascial compartment ('saphenous eye').\n"
        "2. If SFJ competent but thigh GSV shows reflux → Hunterian perforator is the likely EP N1→N2.\n"
        "3. Perforator maneuvers — all three required (Delfrate 2023):\n"
        "   a. Static squeezing (gravitational test)\n"
        "   b. Paranà maneuver (physiological — preferred over squeezing alone)\n"
        "   c. Valsalva (hypertensive test — outward flow = pathological/pathogenic perforator)\n"
        "4. Pathological perforator: outward flow ≥500 ms AND diameter ≥3.5 mm (AVF 2023).\n"
        "5. Watch for N3 above fascia at same level as N2 in compartment — junction is EP N2→N3 (trunk escape).\n"
        "6. Principle: 'No reflux no re-entry' — if GSV reflux persists below escape, another re-entry exists distally."
    ),

    "popliteal_spj": (
        "EXAMINATION PROTOCOL — Popliteal / SPJ (Gianesini 2014 + Delfrate 2023)\n"
        "1. Position: lateral decubitus (left decubitus for right SSV; right decubitus for left SSV).\n"
        "2. BOTH Paranà (active) AND compression/relaxation (passive CR) must be positive simultaneously\n"
        "   to confirm SPJ incompetence (EP N1→N2 at SPJ). One positive alone ≠ incompetence.\n"
        "3. SPJ location is variable — may connect to gastrocnemian vein rather than popliteal vein directly (Delfrate 2023).\n"
        "4. Assess Giacomini vein separately (posterior thigh, SSV→GSV connection).\n"
        "   Forward flow in Giacomini during Paranà = viable outflow route.\n"
        "5. When planning surgery: SPJ disconnection should be performed below the Giacomini junction in mixed shunts."
    ),

    "calf": (
        "EXAMINATION PROTOCOL — Calf (Adler 2022 + Delfrate 2023)\n"
        "1. Track N3 tributaries distally toward re-entry perforators along medial and posterior surfaces.\n"
        "2. Paranà maneuver: inward perforator flow during muscle DIASTOLE (relaxation) = re-entry point (RP N3→N1).\n"
        "   Diastolic reflux into deep system via perforator is always pathological and pathogenic.\n"
        "3. Biphasic perforator: systolic outward flow followed by diastolic inward flow = likely re-entry candidate.\n"
        "   The diastolic inflow is the haemodynamically significant phase (Delfrate 2023).\n"
        "4. Pathological perforator (AVF 2023): outward flow ≥500 ms AND diameter ≥3.5 mm.\n"
        "5. Squeezing alone is insufficient — use Paranà (proprioceptive, physiological) as primary maneuver.\n"
        "6. Medial calf perforators (Boyd/paratibial, posterior tibial) are most relevant for GSV circuit re-entry."
    ),

    "ankle_ssv": (
        "EXAMINATION PROTOCOL — Ankle / Lower Calf (Adler 2022 + Delfrate 2023)\n"
        "1. GSV at medial malleolus (posY 0.85–1.00): medial surface, N2 in fascial compartment.\n"
        "2. SSV at lateral ankle: assess in lateral decubitus. Confirm it joins SPJ posteriorly.\n"
        "3. Paranà squeeze/release at distal perforators: inward flow on release = RP N3→N1 (circuit closure).\n"
        "4. Distal calf SSV assessment is mandatory when stasis ulcers are present (Adler 2022).\n"
        "5. Confirm diameter and outward flow duration to classify perforators as pathological."
    ),

    "general_sequence": (
        "GENERAL EXAMINATION SEQUENCE (Adler 2022 + Delfrate 2023)\n"
        "Step 1 — DVT: compression assessment of all deep veins before any reflux testing.\n"
        "Step 2 — Deep reflux: Valsalva for iliac valve competence; CFV reflux check above SFJ.\n"
        "Step 3 — SFJ: Mickey Mouse sign; Valsalva + Paranà (both must be positive); AASV separately.\n"
        "Step 4 — GSV trunk: medial thigh → calf; saphenous eye in transverse; 500 ms reflux threshold.\n"
        "Step 5 — Perforators: Hunterian zone and calf; all 3 maneuvers; note biphasic flow.\n"
        "Step 6 — SPJ: posterior knee; both Paranà + CR; variable anatomy (check Giacomini).\n"
        "Step 7 — SSV trunk: posterior calf; lateral approach; same 500 ms threshold.\n"
        "Step 8 — Re-entry perforators: identify by diastolic inward flow; confirm ≥3.5 mm diameter.\n"
        "Patient positioning: Reverse Trendelenburg ≥60° for ALL reflux studies (Adler 2022)."
    ),
}


def get_protocol(region: str, pos_y: float) -> str:
    """
    Return the examination protocol for the current probe position.

    posY takes priority over region name for intra-region zone selection
    (e.g. GSV-THI at posY 0.28 returns Hunterian protocol, not upper-thigh).

    Args:
        region:  Anatomical region string (e.g. "SFJ", "GSV-THI", "GSV-CAL", "SPJ", "SSV").
        pos_y:   Probe posY ratio (0.0 = groin, 1.0 = ankle).

    Returns:
        Multi-line protocol string ready to embed in the LLM state message.
    """
    r = region.upper().replace("_", "-")

    # Named junction regions take precedence regardless of posY.
    if r == "SFJ":
        return _PROTOCOLS["sfj_groin"]
    if r == "SPJ":
        return _PROTOCOLS["popliteal_spj"]
    if r == "SSV":
        return _PROTOCOLS["calf"]

    # For all other regions (GSV-THI, GSV-CAL, UNKNOWN, etc.) use posY bands.
    # Boundaries from DuplexUS 2014 p.33-34, Lee 2017 p.129, Adler 2022.
    if pos_y <= 0.07:
        return _PROTOCOLS["sfj_groin"]
    if pos_y <= 0.20:
        return _PROTOCOLS["upper_thigh"]
    if pos_y <= 0.33:
        return _PROTOCOLS["mid_thigh_dodd"]
    if pos_y <= 0.47:
        return _PROTOCOLS["lower_thigh_hunterian"]
    if pos_y <= 0.57:
        return _PROTOCOLS["popliteal_spj"]
    if pos_y <= 0.88:
        return _PROTOCOLS["calf"]
    return _PROTOCOLS["ankle_ssv"]
