"""
Examination Protocol Agent.

Returns the zone-specific duplex ultrasound examination protocol for the
current probe position, sourced from medical literature.

Sources (all verified from PDF reading, June 2026):
  Adler et al. 2022  — RadioGraphics: varicose veins evaluation protocols
  Gianesini et al. 2014 — Phlebology: CHIVA strategy
  Delfrate 2023 — JTAVR: CHIVA duplex assessment protocol
  AVF 2023 guidelines — perforator criteria
  Mendoza et al. 2014 — Duplex Ultrasound of Superficial Leg Veins
    Ch. 7.2  — GSV examination objectives
    Ch. 8.2  — SSV examination objectives
    Ch. 9.2  — Perforating vein examination objectives
    Ch. 10.2 — Tributary examination objectives
    Ch. 14   — Deep leg vein (DVT/compression) assessment
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
        "7. If Valsalva NEGATIVE but Paranà positive: terminal valve is competent — reflux is pre-terminal or from a pelvic leak point. Check SGP, IGP, OP (Delfrate 2023 p.22).\n"
        "8. If BOTH Valsalva AND Paranà positive: SFJ incompetence confirmed (EP N1→N2 at groin) — proceed distally along GSV trunk to establish extent of reflux.\n"
        "9. Assess AASV (anterior accessory saphenous vein) separately — lies anterior to GSV, classified N3 not N2."
    ),

    "upper_thigh": (
        "EXAMINATION PROTOCOL — Upper Thigh / GSV Proximal (Adler 2022)\n"
        "1. Transverse B-mode: confirm GSV sits within 'saphenous eye' (fascial compartment) — N2 identity.\n"
        "2. Measure GSV anteroposterior diameter (document at this level).\n"
        "3. Apply Paranà/squeeze and release: reflux >500 ms = trunk reflux (RP N2→N1).\n"
        "4. AASV may run parallel to GSV in upper thigh — assess it separately (N3, not N2).\n"
        "5. After SFJ entry (EP N1→N2) is confirmed, the next findings to establish are trunk reflux (RP N2→N1) and any trunk-to-tributary escape (EP N2→N3) — both assessed along the medial thigh."
    ),

    "hunterian_proximal": (
        "EXAMINATION PROTOCOL — Proximal Thigh / Hunterian Zone (posY 0.21–0.33) (Adler 2022 + Delfrate 2023)\n"
        "Source: Hunterian perforators = proximal/middle thigh, within Hunter's canal (DuplexUS 2014 p.33)\n"
        "1. Transverse B-mode at medial proximal thigh: confirm GSV in fascial compartment ('saphenous eye').\n"
        "2. KEY ZONE FOR EP N1→N2: Hunterian perforators connect femoral vein (FV) to GSV within Hunter's canal.\n"
        "   If SFJ is competent but thigh GSV shows reflux → Hunterian perforator is the likely EP N1→N2.\n"
        "3. Perforator maneuvers — all three required (Delfrate 2023):\n"
        "   a. Static squeezing (gravitational test)\n"
        "   b. Paranà maneuver (physiological — preferred over squeezing alone)\n"
        "   c. Valsalva (hypertensive test — outward flow = pathological/pathogenic perforator)\n"
        "4. Pathological perforator: outward flow ≥500 ms AND diameter ≥3.5 mm (AVF 2023).\n"
        "5. Watch for N3 above fascia at same level as N2 in compartment — junction is EP N2→N3 (trunk escape).\n"
        "6. Trunk reflux visible here without SFJ entry confirms Hunterian perforator as entry point (EP N1→N2)."
    ),

    "dodd_distal": (
        "EXAMINATION PROTOCOL — Distal Thigh / Dodd Zone (posY 0.34–0.47) (Adler 2022 + Delfrate 2023)\n"
        "Source: Dodd perforators = distal third of thigh, just above the knee (DuplexUS 2014 p.33)\n"
        "1. Transverse B-mode at medial distal thigh: confirm GSV in fascial compartment ('saphenous eye').\n"
        "2. Dodd perforators connect the femoral vein (FV) to the GSV just above the knee.\n"
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
        "6. Medial calf perforators (paratibial, posterior tibial) are the most common GSV re-entry sites (Mendoza 2014 Ch. 9.2; Delfrate 2023)."
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


def get_protocol(region: str, pos_y: float, vein_mode: str = "") -> str:
    """
    Return the examination protocol for the current probe position.

    posY takes priority over region name for intra-region zone selection
    (e.g. GSV-THI at posY 0.28 returns Hunterian protocol, not upper-thigh).
    When vein_mode is set, the vein-specific examination objectives from
    Mendoza 2014 are appended to give comprehensive examination guidance.

    Args:
        region:    Anatomical region string (e.g. "SFJ", "GSV-THI", "GSV-CAL", "SPJ", "SSV").
        pos_y:     Probe posY ratio (0.0 = groin, 1.0 = ankle).
        vein_mode: Active vein scan mode ("GSV", "SSV", "PERFORATORS",
                   "TRIBUTARIES", "DEEP_VEINS", or "" for none).

    Returns:
        Multi-line protocol string ready to embed in the LLM state message.
    """
    r = region.upper().replace("_", "-")

    # Named junction regions take precedence regardless of posY.
    if r == "SFJ":
        zone_proto = _PROTOCOLS["sfj_groin"]
    elif r == "SPJ":
        zone_proto = _PROTOCOLS["popliteal_spj"]
    elif r == "SSV":
        zone_proto = _PROTOCOLS["calf"]
    # For all other regions (GSV-THI, GSV-CAL, UNKNOWN, etc.) use posY bands.
    # Boundaries from DuplexUS 2014 p.33-34, Adler 2022.
    elif pos_y <= 0.07:
        zone_proto = _PROTOCOLS["sfj_groin"]
    elif pos_y <= 0.20:
        zone_proto = _PROTOCOLS["upper_thigh"]
    elif pos_y <= 0.33:
        zone_proto = _PROTOCOLS["hunterian_proximal"]
    elif pos_y <= 0.47:
        zone_proto = _PROTOCOLS["dodd_distal"]
    elif pos_y <= 0.57:
        zone_proto = _PROTOCOLS["popliteal_spj"]
    elif pos_y <= 0.88:
        zone_proto = _PROTOCOLS["calf"]
    else:
        zone_proto = _PROTOCOLS["ankle_ssv"]

    vein_obj = get_vein_examination_protocol(vein_mode)
    if vein_obj:
        return zone_proto + "\n\n" + vein_obj
    return zone_proto


# ─────────────────────────────────────────────────────────────────────────────
# VEIN-SPECIFIC EXAMINATION OBJECTIVES (Mendoza et al. 2014, Chapters 7-10, 14)
#
# When the operator declares they are examining a specific vein ("scan GSV",
# "scan SSV", etc.), these comprehensive checklists from the Mendoza book are
# appended to the zone protocol so the LLM can guide the examiner through
# ALL clinically relevant questions for that vein — not just positional maneuvers.
# ─────────────────────────────────────────────────────────────────────────────

_VEIN_EXAM_OBJECTIVES: dict[str, str] = {

    "GSV": (
        "VEIN EXAMINATION OBJECTIVES — GSV (Mendoza 2014, Ch. 7.2)\n"
        "The following questions must be answered during a complete GSV examination:\n"
        "JUNCTION (SFJ):\n"
        "  1. Is there reflux through the saphenofemoral junction (SFJ)?\n"
        "  2. Is the TERMINAL valve competent?\n"
        "  3. Is the PRE-TERMINAL valve competent?\n"
        "  4. Are the superficial inguinal veins (AASV, PASV, pudendal, epigastric, SCIV) competent?\n"
        "  5. Is there any anatomical anomaly at the SFJ (accessory confluences, duplications)?\n"
        "  6. Is a venous ANEURYSM present at the junction?\n"
        "TRUNK COURSE (groin → ankle):\n"
        "  7. Is the GSV visible in its FASCIAL COMPARTMENT ('saphenous eye') throughout its length?\n"
        "  8. Is the entire course INTERFASCIAL (confirm in transverse view at each level)?\n"
        "  9. Measure GSV DIAMETER at the standardised point 10 cm below the SFJ.\n"
        " 10. Are there any SUDDEN CHANGES in saphenous calibre (segmental aplasia, duplication)?\n"
        " 11. Is the course TYPICAL relative to topographical anatomy, or more lateral than usual?\n"
        " 12. Is there a DUPLICATION along part of the course?\n"
        " 13. Is there an APLASTIC or HYPOPLASTIC segment?\n"
        " 14. Is there evidence of SUPERFICIAL VEIN THROMBOSIS or post-phlebitic changes in the wall?\n"
        "REFLUX ASSESSMENT:\n"
        " 15. Is there REFLUX in the GSV? (Use Paranà + Valsalva; threshold >500 ms.)\n"
        " 16. Is this reflux WELL or POORLY drained? (Assess PW reflux curve profile.)\n"
        " 17. How many saphenous SEGMENTS are refluxive? (May have competent segments between refluxive ones.)\n"
        " 18. Where is the PROXIMAL REFLUX SOURCE? (SFJ, Hunterian perforator, Dodd perforator, pelvic?)\n"
        "ESCAPE AND RE-ENTRY:\n"
        " 19. Are there DILATED TRIBUTARIES or PERFORATING VEINS along the GSV course?\n"
        " 20. Are they involved in a RECIRCULATION CIRCUIT?\n"
        " 21. Does the reflux LEAVE THE GSV via a tributary or a perforating vein (escape point)?\n"
        "CHIVA CLASSIFICATION:\n"
        " 22. What is the HACH CLASS (I–IV) where applicable?\n"
        " 23. Which CHIVA SHUNT TYPE (1, 2A, 2B, 2C, 3, 1+2, 4, 5, 6) is present?\n"
        "DISTAL SEGMENT:\n"
        " 24. How does the CALIBRE AND COURSE of the GSV behave DISTAL to the end of reflux?\n"
        "SURROUNDING TISSUE:\n"
        " 25. Are there any PATHOLOGICAL SOFT TISSUE CHANGES surrounding the GSV?\n"
        "SECONDARY VARICES:\n"
        " 26. Is the flow in the dilated GSV ANTEGRADE because it is serving as drainage for "
        "obstructed deep veins? (Secondary varicose veins — assess deep system if suspected.)\n"
        "PRIOR TREATMENT:\n"
        " 27. Has all or part of the saphenous vein been PREVIOUSLY TREATED (ablation/stripping)?"
    ),

    "SSV": (
        "VEIN EXAMINATION OBJECTIVES — SSV (Mendoza 2014, Ch. 8.2)\n"
        "The following questions must be answered during a complete SSV examination:\n"
        "JUNCTION (SPJ):\n"
        "  1. Does the SSV JOIN THE POPLITEAL VEIN (SPJ present)?\n"
        "  2. At what LEVEL is the SPJ relative to the POSTERIOR KNEE CREASE? "
        "(54% are above crease; high junctions lie under distal thigh musculature.)\n"
        "  3. Is the SSV CONNECTED TO MUSCLE VEINS (gastrocnemius veins) in the junction region?\n"
        "THIGH EXTENSION:\n"
        "  4. Does a THIGH EXTENSION of the SSV exist above the popliteal fossa?\n"
        "  5. Does it form a GIACOMINI VEIN (femoropopliteal vein connecting to GSV)?\n"
        "  6. Is the thigh extension or the Giacomini vein COMPETENT? "
        "(Physiological flow in Giacomini is DOWNWARD into SSV — upward = reflux.)\n"
        "  7. Are the MUSCLE VEINS (gastrocnemius, soleus) at the junction competent?\n"
        "  8. Is there any ANATOMICAL ANOMALY at the SSV junction (absent SPJ, direct muscle vein junction)?\n"
        "  9. Is the SPJ REFLUXIVE, DILATED or ANEURYSMAL?\n"
        "TRUNK COURSE (popliteal fossa → lateral ankle):\n"
        " 10. Is the WHOLE COURSE of the SSV INTERFASCIAL within its fascial compartment?\n"
        " 11. Measure SSV DIAMETER at the standardised point 5 cm below the SPJ.\n"
        " 12. Are there any SUDDEN CHANGES IN CALIBRE?\n"
        " 13. Is the SSV course TYPICAL, or does it DEVIATE or DUPLICATE?\n"
        " 14. Does any segment have SUPERFICIAL VEIN THROMBOSIS or post-thrombotic alteration?\n"
        "REFLUX ASSESSMENT:\n"
        " 15. Is there REFLUX in the SSV? (Use BOTH Paranà AND compression/relaxation; "
        "BOTH must be positive to confirm SPJ incompetence.)\n"
        " 16. Is this reflux WELL or POORLY drained? (PW reflux curve profile.)\n"
        " 17. How many SSV SEGMENTS are refluxive?\n"
        " 18. Where is the PROXIMAL REFLUX SOURCE? (SPJ, muscle vein, Giacomini re-fill from GSV?)\n"
        "ESCAPE AND RE-ENTRY:\n"
        " 19. Are there DILATED TRIBUTARIES or PERFORATING VEINS along the SSV course?\n"
        " 20. Are they involved in a RECIRCULATION LOOP?\n"
        " 21. Does the reflux LEAVE THE SSV via a tributary or perforating vein (escape point)?\n"
        " 22. How long is the REFLUXIVE SEGMENT?\n"
        "CHIVA CLASSIFICATION:\n"
        " 23. Which CHIVA SHUNT TYPE is found?\n"
        "DISTAL SEGMENT:\n"
        " 24. How does the CALIBRE AND COURSE of the SSV behave BELOW the end of reflux?\n"
        "SURROUNDING TISSUE:\n"
        " 25. Are there any REMARKABLE FINDINGS in the tissues surrounding the SSV?\n"
        "SECONDARY VARICES:\n"
        " 26. Is the flow in the dilated SSV ANTEGRADE because it is serving as "
        "DRAINAGE FOR THE DEEP SYSTEM? (Secondary varices — assess deep system.)"
    ),

    "PERFORATORS": (
        "VEIN EXAMINATION OBJECTIVES — PERFORATING VEINS (Mendoza 2014, Ch. 9.2)\n"
        "Screen questions (all perforating veins):\n"
        "  1. Are there DILATED PERFORATING VEINS in the courses of the GSV and SSV?\n"
        "  2. Are there visible perforating veins in OTHER TYPICAL LOCATIONS: "
        "posterior arch vein, back/lateral thigh, back/lateral calf?\n"
        "  3. If found: is it a MUSCLE PERFORATING VEIN (connects deep vein via muscle vein "
        "to superficial vein)? Is there a POST-TRAUMATIC cause for dilation?\n"
        "  4. Have the perforating veins suffered POST-THROMBOTIC DAMAGE (wall irregularity)?\n"
        "  5. What is their DIAMETER? (Only perforators >3 mm in B-scan need Doppler assessment.)\n"
        "For every perforating vein with diameter >3 mm — determine its haemodynamic role:\n"
        "  6. Does it form the UPPER END OF RECIRCULATION? "
        "(Reflux source: diastolic outward flow is the defining criterion.)\n"
        "  7. Does it form the LOWER END OF RECIRCULATION? "
        "(Re-entry point: diastolic INWARD flow from refluxive superficial vein.)\n"
        "  8. Does it present a BLOWOUT? (Outward bulging during Valsalva.)\n"
        "  9. Is the perforating vein located UNDER AN ULCER?\n"
        "Examination technique for each dilated perforator:\n"
        "  — Assess in DIASTOLE (after muscle relaxation): diastolic outward flow = reflux source; "
        "diastolic inward flow = re-entry point.\n"
        "  — Apply ALL THREE maneuvers (Delfrate 2023): "
        "(a) Static squeezing, (b) Paranà maneuver, (c) Valsalva.\n"
        "  — Pathological threshold (AVF 2023): outward flow ≥500 ms AND diameter ≥3.5 mm.\n"
        "  — Biphasic pattern (outward systolic + inward diastolic) = likely re-entry candidate.\n"
        "  — DO NOT apply tourniquet above perforator during assessment: occludes recirculation "
        "and may produce false pathological result.\n"
        "Key anatomical sites to examine:\n"
        "  Hunterian (proximal thigh, posY 0.21-0.33): key EP N1→N2 when SFJ competent.\n"
        "  Dodd (distal thigh, posY 0.34-0.47): joins FV to GSV just above knee.\n"
        "  Paratibial/Boyd (upper calf medial): may act as reflux source for GSV.\n"
        "  Posterior tibial / Cockett group (calf medial): common re-entry perforators.\n"
        "  Posterior thigh perforators: may fill Giacomini or varicose tributaries on back of thigh.\n"
        "  Gastrocnemius perforators (medial/lateral calf): connect SSV circuit."
    ),

    "TRIBUTARIES": (
        "VEIN EXAMINATION OBJECTIVES — TRIBUTARIES (Mendoza 2014, Ch. 10.2)\n"
        "The following questions must be answered for each refluxive tributary:\n"
        "  1. Is the TRIBUTARY REFLUXIVE? (Confirm with PW Doppler — flow >500 ms after provocation.)\n"
        "  2. WHERE DOES IT FILL WITH REFLUX? (Which saphenous trunk or perforator feeds it?)\n"
        "  3. Does the tributary have other INFLOW POINTS? "
        "(Perforating veins, saphenous veins, or other tributaries draining into it.)\n"
        "  4. Does the tributary FILL OTHER TRIBUTARIES REFLUXIVELY?\n"
        "  5. Does the tributary FILL A SAPHENOUS VEIN REFLUXIVELY? "
        "(Ascending reflux from tributary into competent saphenous vein — siphon effect.)\n"
        "  6. What are the DRAINAGE POINTS? "
        "(Perforating veins as re-entry, or direct drainage into saphenous trunks.)\n"
        "  7. Is the tributary WELL or POORLY DRAINED? "
        "(Well drained = fast-onset, short-duration PW curve; poorly drained = slow prolonged curve.)\n"
        "  8. Is the tributary affected by SUPERFICIAL VEIN THROMBOSIS (phlebitis)?\n"
        "  9. Is it a SECONDARY VARICOSE VEIN serving as DRAINAGE into the deep veins "
        "(i.e. antegrade flow draining obstructed deep system)?\n"
        "Systematic examination approach:\n"
        "  — Distinguish tributary from saphenous trunk: tributary runs ABOVE the saphenous fascia; "
        "saphenous trunk runs WITHIN the fascial compartment.\n"
        "  — Always examine: AASV (anterior accessory saphenous vein), PASV (posterior accessory), "
        "posterior arch vein (runs along medial calf), and Giacomini/femoropopliteal vein.\n"
        "  — For DRAINAGE POINT location: compress tributary at progressively distal points "
        "while applying Wunstorf (toe-raise) maneuver — reflux onset shift identifies drainage. \n"
        "  — For REFLUX SOURCE identification: place probe on segment, colour duplex mode, "
        "tap vein proximally — if signal appears, reflux source is further upstream.\n"
        "AASV-specific notes:\n"
        "  — First 15-20 cm of AASV runs interfascially (same or adjacent compartment to GSV). "
        "After that it becomes epifascial. Examine throughout its course regardless of junction competence.\n"
        "  — Common drainage: lateral knee perforators; anterior tributary above patella; "
        "rarely medial tributary parallel to GSV.\n"
        "Posterior arch vein notes:\n"
        "  — Runs from medial malleolus up medial calf, collecting posterior calf tributaries.\n"
        "  — Perforators along its course (Cockett group) are the most common re-entry perforators.\n"
        "Giacomini / femoropopliteal vein notes:\n"
        "  — Present in 65% of legs as cranial extension of SSV.\n"
        "  — Physiological flow: DOWNWARD (away from head, into SSV).\n"
        "  — If reflux from GSV fills SSV via Giacomini → SSV cannot be treated without GSV."
    ),

    "DEEP_VEINS": (
        "VEIN EXAMINATION OBJECTIVES — DEEP VEINS (Mendoza 2014, Ch. 14)\n"
        "DVT exclusion is MANDATORY before any reflux assessment of superficial veins.\n"
        "Examination sequence (complete compression ultrasound — CCUS):\n"
        "  1. PATIENT POSITION: standing (veins distend and are easier to find) AND supine "
        "(for compression — muscle tension in standing can resist compression).\n"
        "  2. Examine the CONTRALATERAL LEG: co-existing DVT found in up to 20% of cases.\n"
        "PROXIMAL DEEP VEINS (thigh):\n"
        "  3. COMMON FEMORAL VEIN (CFV) at groin: transverse compression with B-scan. "
        "Complete compressibility of the lumen = DVT excluded at this level.\n"
        "  4. FEMORAL VEIN (FV) along medial thigh (formerly 'superficial femoral vein' — term deprecated): "
        "scan entire length in transverse. NOTE: difficult to compress at adductor canal — use longitudinal view.\n"
        "  5. DEEP FEMORAL VEIN (DFV): visible only at its junction with FV from posterolateral aspect.\n"
        "  6. POPLITEAL VEIN (PV) in popliteal fossa: transverse compression. "
        "Variable anatomy — may be duplicated; always compress both lumina if duplication present.\n"
        "CALF DEEP VEINS:\n"
        "  7. POSTERIOR TIBIAL VEINS: paired, medial calf, best seen from medial approach. "
        "Superficial in lower third — easier to find in standing patient.\n"
        "  8. PERONEAL VEINS: posterior to interosseous membrane, lateral aspect. "
        "Best seen in upper calf where they form larger calibre vessels.\n"
        "  9. ANTERIOR TIBIAL VEINS: anterior to interosseous membrane; small calibre. "
        "Probe lateral to edge of tibia in transverse view.\n"
        " 10. MUSCLE VEINS — GASTROCNEMIUS veins (join popliteal vein at popliteal fossa) "
        "and SOLEUS veins: must be included in routine protocol (frequent DVT origin).\n"
        "Compression ultrasound criteria:\n"
        "  — EXCLUSION CRITERION: complete compressibility of vein lumen in B-scan transverse view.\n"
        "  — Colour signal disappearance under pressure is supportive but NOT a strict requirement.\n"
        "  — Fresh thrombus may still compress (soft material) — very fresh DVT can be missed.\n"
        "  — Non-thrombosed vein segments in strong muscle (adductor canal, peroneal) may appear "
        "uncompressible — verify by changing probe angle or using longitudinal view.\n"
        "  — Always examine SFJ and SPJ in LONGITUDINAL view to check for ascending thrombus "
        "from superficial veins emerging into the deep system.\n"
        "Post-thrombotic assessment (when DVT history known):\n"
        "  — Check for wall irregularity, partial recanalisation, residual thrombus.\n"
        "  — Reflux threshold in deep veins: >1 second (vs >500 ms for superficial veins).\n"
        "  — Assess femoral and popliteal veins in LONGITUDINAL view for post-thrombotic reflux "
        "using Valsalva (groin valves) and calf compression/relaxation (more distal valves).\n"
        "  — Elongation and tortuosity of popliteal vein = secondary deep vein insufficiency "
        "(reversible after superficial vein treatment) — distinguish from post-thrombotic syndrome."
    ),
}


def get_vein_examination_protocol(vein_mode: str) -> str:
    """
    Return the comprehensive vein-specific examination objectives from Mendoza 2014.

    Called when the operator has declared a specific vein scan mode (e.g. after
    saying 'scan GSV'). The returned text is appended to the zone protocol so the
    LLM has full context on what to verify during this vein examination.

    Args:
        vein_mode: "GSV", "SSV", "PERFORATORS", "TRIBUTARIES", "DEEP_VEINS", or "".

    Returns:
        Examination objectives string, or "" if vein_mode is unrecognised.
    """
    key = vein_mode.upper().strip()
    return _VEIN_EXAM_OBJECTIVES.get(key, "")