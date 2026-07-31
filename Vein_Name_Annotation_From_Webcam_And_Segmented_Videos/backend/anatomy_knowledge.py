"""
Anatomy knowledge fed to the VLM as reasoning context for Stage 3 (webcam location +
vein naming). This is prose the model reasons over, never a lookup table branched on in
Python — see the project plan for why (Task_2_App has three separate, drifting hardcoded
region->vein tables; this project exists partly to avoid repeating that).

Distilled from: Lee et al. 2017 (Ultrasonography), Adler et al. 2022 (RadioGraphics),
Gianesini/Zamboni/Mendoza's "Saphenous Vein Sparing Strategies in Chronic Venous Disease"
2018, Caggiati et al.'s "Duplex Ultrasound of Superficial Leg Veins" 2014, Almeida's
"Atlas of Endovascular Venous Surgery", Gloviczki's "Handbook of Venous and Lymphatic
Disorders". Exact perforator cm-measurements vary across the historical studies cited in
these books — treat any specific cm figure below as approximate, not consensus.
"""

LEG_LEVELS = [
    "groin_sfj",
    "upper_thigh",
    "proximal_thigh_hunterian",
    "distal_thigh_dodd",
    "knee_popliteal",
    "calf",
    "ankle",
    "uncertain",
]

ANATOMY_REFERENCE_TEXT = """
FASCIAL DEPTH — why N1/N2/N3 exist anatomically:
Two fasciae define three compartments on ultrasound. The muscle fascia (deep) is the
floor. The saphenous fascia (a condensed layer of subcutaneous tissue) is the roof — it
only exists as a distinct sheath where it separates from the muscle fascia to wrap a
saphenous trunk, forming the "saphenous compartment". This is the classic "saphenous eye"
/ "Egyptian eye" ultrasound sign: upper lid = saphenous fascia, lower lid = muscle fascia,
iris = the saphenous vein (GSV or SSV) inside.
- N1 (subfascial/deep): below the muscle fascia — femoral vein, popliteal vein, tibial
  and peroneal veins.
- N2 (interfascial/saphenous compartment): between the two fasciae — GSV, SSV, and only
  the proximal/interfascial portions of AASV, PASV, and the Giacomini vein.
- N3 (epifascial/superficial): above the saphenous fascia, in subcutaneous fat —
  tributaries, varicosities, reticular veins, and AASV/PASV once they leave the compartment.

LEG LEVELS, GROIN TO ANKLE:

groin_sfj — Groin / saphenofemoral junction: GSV joins the common femoral vein (CFV,
deep/N1) at the groin crease. Tributaries converge here: superficial epigastric,
superficial circumflex iliac, superficial external pudendal veins, plus the anterior and
posterior accessory saphenous veins (AASV/PASV) joining a few cm distal. Expect CFV (N1),
GSV (N2), and tributaries (N3) all close together.

upper_thigh — Upper/mid thigh, front-medial surface: GSV runs in its compartment (the
"saphenous eye") on the medial thigh; the femoral vein runs deep, separated from GSV by
the sartorius muscle around mid-thigh. AASV, if present, runs parallel and slightly
lateral/anterior to GSV — interfascial (N2) proximally, becoming epifascial (N3) at a
variable point further down.

proximal_thigh_hunterian — Border of the middle/lower thigh thirds: the Hunterian
perforator zone (perforating veins of the adductor canal) — a direct GSV-to-femoral-vein
connection, one of the few perforators that joins the trunk directly rather than via a
tributary.

distal_thigh_dodd — Distal thigh, just above the knee: the Dodd perforator zone — another
femoral-vein-to-GSV connection, usually below the adductor canal itself.

knee_popliteal — Knee / popliteal fossa: anatomy here is genuinely variable, not a fixed
point — only around 60% of legs have the "typical" SPJ (small saphenous vein joining the
popliteal vein directly in the popliteal fossa); much of the remainder drains via the
Giacomini vein into the GSV, or via a thigh perforator into the deep femoral vein, instead
of (or alongside) a popliteal-fossa junction. No fascia separates the SPJ from the
popliteal vein (unlike the SFJ, which has an intervening cribriform fascia). Expect SSV
(N2) approaching from below/behind, popliteal vein (N1) deep, and possibly the Giacomini
vein (N2 while interfascial, becoming N3 more proximally) running up the posterior thigh.

calf, front/medial surface: GSV continues down the medial calf in its compartment; the
posterior arch vein ("Leonardo's vein") runs a few cm posterior to it along "Linton's
line" and is where most pathological calf perforators are found — Boyd's (paratibial,
just below the knee) and the Cockett group (posterior tibial perforators, lower-to-mid
medial calf). Deep veins here: paired posterior tibial and peroneal veins.

calf, back/posterior surface: SSV runs interfascially the entire way from ankle to
popliteal fossa (a common but incorrect claim is that it becomes superficial partway down
— it does not). Several fairly constant SSV-to-deep perforators exist along this course
(soleal/"gastrocnemius point", para-Achillean, lateral ankle).

ankle — GSV originates from the medial end of the dorsal venous arch of the foot, visible
just anterior to the medial malleolus. SSV originates from the lateral end, visible just
posterior to the lateral malleolus. Both compartments are at their narrowest here.

GENERAL REASONING NOTES:
- Front/medial surface + thigh or calf level -> GSV territory: N2 = GSV, N1 = femoral vein
  (CFV specifically only at the groin).
- Back/posterior surface + calf level, or the popliteal fossa -> SSV territory: N2 = SSV,
  N1 = popliteal vein.
- An N3 blob is a tributary, an accessory vein once epifascial (AASV/PASV), or a
  varicosity — name it generically as "Tributary" unless the location evidence points
  specifically to AASV/PASV (e.g. clearly at the SFJ or running parallel to GSV where
  those veins are expected).
- Do not force a confident specific vein name when the leg-location evidence is weak or
  "uncertain" — say so. A wrong specific name is worse than an honest "GSV or tributary,
  location uncertain".
""".strip()
