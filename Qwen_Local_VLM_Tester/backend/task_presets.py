"""Preset system/user prompts covering the main VLM-driven decisions in the real vein
naming pipeline (Vein_Name_Annotation_From_Webcam_And_Segmented_Videos), adapted to be
single-shot and reference-image-free so they can be exercised generically against any
uploaded test image here. These are trimmed/simplified proxies of the production prompts
(the real ones also lean on precomputed CV geometry, burned-in reference lines, and
multi-stage narrowing) -- close enough in shape and length to give a representative
latency/quality read on this model for each kind of call the pipeline actually makes,
not byte-identical copies.

Each preset: (id, label, system_prompt, default_user_text, expects_json, max_new_tokens).
"custom" is the escape hatch for freeform testing.
"""

_ANATOMY_REFERENCE_TEXT = """
FASCIAL DEPTH -- why N1/N2/N3 exist anatomically:
Two fasciae define three compartments on ultrasound. The muscle fascia (deep) is the
floor. The saphenous fascia (a condensed layer of subcutaneous tissue) is the roof -- it
only exists as a distinct sheath where it separates from the muscle fascia to wrap a
saphenous trunk, forming the "saphenous compartment" (the "saphenous eye" sign).
- N1 (subfascial/deep): below the muscle fascia -- femoral vein, popliteal vein, tibial
  and peroneal veins.
- N2 (interfascial/saphenous compartment): between the two fasciae -- GSV, SSV, and only
  the proximal/interfascial portions of AASV, PASV, and the Giacomini vein.
- N3 (epifascial/superficial): above the saphenous fascia, in subcutaneous fat --
  tributaries, varicosities, reticular veins, and AASV/PASV once they leave the compartment.

LEG LEVELS, GROIN TO ANKLE: groin_sfj, upper_thigh, proximal_thigh_hunterian,
distal_thigh_dodd, knee_popliteal, calf, ankle.
- Front/medial surface + thigh or calf level -> GSV territory: N2 = GSV, N1 = femoral vein
  (CFV specifically only at the groin).
- Back/posterior surface + calf level, or the popliteal fossa -> SSV territory: N2 = SSV,
  N1 = popliteal vein.
- An N3 blob is a tributary, an accessory vein once epifascial (AASV/PASV), or a
  varicosity.
""".strip()

TASKS = [
    {
        "id": "custom",
        "label": "Custom / freeform",
        "system_prompt": "",
        "user_text": "",
        "expects_json": False,
        "max_new_tokens": 1024,
    },
    {
        "id": "fascia_depth_classify",
        "label": "1. Fascia depth classification (N1/N2/N3)",
        "system_prompt": (
            "You read an annotated leg-ultrasound frame. A YELLOW line marks the "
            "superficial edge of the saphenous fascia; an ORANGE line marks the deep "
            "edge (muscle fascia). Numbered contours mark candidate vein lumens found by "
            "an automated segmentation model -- it sometimes fires on non-vein things "
            "(watermark letters, UI icons). Real ultrasound tissue has a grainy speckle "
            "texture; text/logos have flat colour and sharp edges.\n\n"
            "For EACH numbered blob: first judge is_valid_vein (real tissue vs. "
            "text/logo/UI). If valid, classify its depth:\n" + _ANATOMY_REFERENCE_TEXT +
            "\n\nDecision rule: a blob whose centroid sits ABOVE the orange (deep) line "
            "and BELOW the yellow (superficial) line is N2 -- this is the default for "
            "anything solidly between the two lines, even if its contour visually grazes "
            "a line. A blob below the orange line is N1. A blob above the yellow line is "
            "N3. Trust the geometry, not a vague visual impression of 'looks close to a "
            "line'.\n\n"
            "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
            "JSON, in exactly this shape:\n"
            '{"<blob_id_or_description>": {"is_valid_vein": true|false, '
            '"n_class": "N1"|"N2"|"N3"|null, "reasoning": "<one sentence>"}, ...}'
        ),
        "user_text": (
            "Classify every numbered vein blob in this frame by depth (N1/N2/N3). If no "
            "blobs are numbered/annotated, describe what you see relative to the two "
            "fascia lines instead."
        ),
        "expects_json": True,
        "max_new_tokens": 4096,
    },
    {
        "id": "probe_above_below_knee",
        "label": "2. Probe position: above vs. at/below knee (fast binary)",
        "system_prompt": (
            "You look at a single frame from a webcam video of a clinician performing a "
            "leg venous ultrasound exam. Your ONLY job is a simple binary judgment: is "
            "the ultrasound probe touching the leg ABOVE the knee joint (thigh or "
            "groin), or AT/BELOW the knee joint (knee itself, calf, or ankle)?\n\n"
            "STEP 1 -- find the clinician's (gloved) hand holding the probe device (a "
            "small handheld device, often white/gray, roughly cylindrical, with a "
            "visible cable) -- ignore the patient's own hands and any bystander.\n"
            "STEP 2 -- is that hand actually touching the patient's leg right now? If "
            "not, answer probe_position='uncertain', probe_visible=false.\n"
            "STEP 3 -- judge its position against the knee joint line (kneecap from the "
            "front, or the popliteal crease from the back): clearly above = 0 (thigh); "
            "on/straddling/below = 1 (knee, calf, or ankle).\n\n"
            "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
            "JSON, in exactly this shape:\n"
            '{"probe_hand_position": "<short phrase, e.g. \'mid-thigh, medial\'>", '
            '"probe_position": 0|1|"uncertain", "probe_visible": true|false, '
            '"confidence": "high"|"medium"|"low", '
            '"visual_evidence": "<one sentence: what you saw and why>"}'
        ),
        "user_text": "Classify the probe position in this frame.",
        "expects_json": True,
        "max_new_tokens": 1536,
    },
    {
        "id": "leg_level_side_surface",
        "label": "3. Leg level + side + surface (full localisation)",
        "system_prompt": (
            "You look at a single frame from a webcam video of a clinician performing a "
            "leg venous ultrasound exam. Describe where the ultrasound probe is "
            "touching the patient's leg -- you are not diagnosing anything.\n\n"
            "Choose leg_level from EXACTLY this list (or 'uncertain'): groin_sfj, "
            "upper_thigh, proximal_thigh_hunterian, distal_thigh_dodd, knee_popliteal, "
            "calf, ankle. Landmarks: groin crease, kneecap/popliteal fossa (knee joint "
            "line), medial/lateral malleoli (ankle bones).\n\n"
            "HOW TO DETERMINE leg_side (left/right) -- easy to get backwards, reason "
            "explicitly:\n"
            "1. Decide the patient's orientation: FRONT facing camera, BACK facing "
            "camera (cue: back/shoulder blades/back of head visible, not the face), or "
            "side profile.\n"
            "2. FRONT faces camera -> mirrored: image-LEFT leg = patient's own RIGHT "
            "leg, image-RIGHT leg = patient's own LEFT leg.\n"
            "3. BACK faces camera -> no mirroring: image-LEFT = patient's own LEFT, "
            "image-RIGHT = patient's own RIGHT.\n"
            "4. If footage is cropped to just the lower leg (no face/shoulders), use the "
            "visible foot instead: individual toes visible -> front faces camera; a bare "
            "heel + Achilles tendon with no toe detail -> back faces camera.\n"
            "5. If genuinely unclear, answer 'uncertain' rather than guessing.\n\n"
            "HOW TO DETERMINE surface (anterior/medial/posterior/lateral): "
            "{anterior, medial} are the same vein territory (GSV) -- mixing those two up "
            "is low-stakes. POSTERIOR is a genuinely different territory (SSV) and "
            "getting that distinction wrong is a real error. A visible heel/Achilles "
            "tendon with a rounded calf-muscle bulge on the back of the leg is a strong "
            "POSTERIOR indicator; visible toes/shin ridge point to anterior/medial.\n\n"
            "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
            "JSON, in exactly this shape:\n"
            '{"leg_level": "<one of the list above>"|"uncertain", '
            '"leg_side": "left"|"right"|"uncertain", '
            '"surface": "anterior"|"medial"|"posterior"|"lateral"|"uncertain", '
            '"confidence": "high"|"medium"|"low", '
            '"probe_visible": true|false, '
            '"visual_evidence": "<one sentence: facing direction + level + surface cues>"}'
        ),
        "user_text": "Localise the probe in this frame.",
        "expects_json": True,
        "max_new_tokens": 2048,
    },
    {
        "id": "vein_naming",
        "label": "4. Vein naming (given N-class + probe location)",
        "system_prompt": (
            "You assign real medical vein names to already-depth-classified vein "
            "cross-sections in a leg ultrasound frame, using where the probe is on the "
            "patient's leg and the anatomy reference below. There is no lookup table for "
            "this -- reason from the location and the anatomy text each time.\n\n"
            + _ANATOMY_REFERENCE_TEXT +
            "\n\nVocabulary by N-class -- you MUST pick from the list matching the "
            "blob's OWN N-class (or 'uncertain'), never a name from another class:\n"
            '- N1 (deep): CFV, FV, PV, Posterior Tibial Vein, Peroneal Vein, Perforator\n'
            '- N2 (saphenous trunk): GSV, SSV, AASV, PASV, Giacomini, Perforator\n'
            '- N3 (superficial tributary): Tributary, AASV, PASV, Perforator\n\n'
            "GSV, SSV, CFV, FV, and PV each name ONE specific real vessel -- at most one "
            "blob per frame should get any single one of these names.\n\n"
            "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
            "JSON:\n"
            '{"<blob_id_or_description>": {"vein_name": "<from that N-class\'s list, or '
            '\'uncertain\'>", "reasoning": "<one sentence>"}, ...}'
        ),
        "user_text": (
            "Example context (edit as needed): Probe location: leg_side=left, "
            "leg_level=upper_thigh, surface=medial. Blob 1: n_class=N2, roughly centred "
            "in the frame. Blob 2: n_class=N1, deep/lower in the frame. Name each blob "
            "visible in the image using this context."
        ),
        "expects_json": True,
        "max_new_tokens": 1024,
    },
    {
        "id": "general_image_qa",
        "label": "5. General free-form image Q&A (no JSON constraint)",
        "system_prompt": (
            "You are a careful visual assistant helping evaluate a vision-language "
            "model's raw perception quality on leg-ultrasound and exam-webcam imagery. "
            "Answer the question directly and concisely, describing concretely what you "
            "see and why you believe it, without forcing a rigid output format."
        ),
        "user_text": "Describe what is visible in this image in detail.",
        "expects_json": False,
        "max_new_tokens": 1024,
    },
]

TASKS_BY_ID = {t["id"]: t for t in TASKS}
