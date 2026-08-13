"""
Stage 3b: combine the N-classed blobs (Stage 2) and the probe location (Stage 3a) with
the anatomy reference text to assign real medical vein names. VLM-only decision — Python
never branches on leg_level/n_class to pick a name.

Vocabulary gating (per-blob allowed vein names, keyed by that blob's own N-class) mirrors
the SAME narrowed-vocabulary pattern already proven in stage3_webcam_location.py (Stage
A's 0/1 above/below-knee split narrows Stage B's leg_level choices) — see
anatomy_knowledge.VEIN_NAMES_BY_NCLASS for why this exists: without it, nothing stopped
the model naming an N2 (saphenous-trunk) blob "Tributary" (an N3-only concept), a
confirmed real bug. Python validates the model's answer is actually IN the vocabulary it
was given (same structural-validity role as stage3_webcam_location.normalize()'s
allowed_levels check) and retries on a violation — it never itself picks the replacement
name, that's still always the model's call.
"""
import base64

import cv2

import anatomy_knowledge
import groq_client


def _allowed_names_for(n_class: str | None) -> list[str]:
    """Blobs with no N-class (Stage 2 couldn't determine depth) get no vocabulary at all
    — see name_veins(), such blobs are filtered out before ever reaching this module.
    Defensive fallback here (union of all three lists) only matters if that filtering is
    ever bypassed by a future caller."""
    return anatomy_knowledge.VEIN_NAMES_BY_NCLASS.get(
        n_class,
        sorted(set(anatomy_knowledge.N1_VEIN_NAMES + anatomy_knowledge.N2_VEIN_NAMES + anatomy_knowledge.N3_VEIN_NAMES),
        )
    )


def _system_prompt(anatomy_text: str) -> str:
    return (
        "You assign real medical vein names to already-depth-classified vein "
        "cross-sections in a leg ultrasound frame, using where the probe is on the "
        "patient's leg (from a paired webcam frame) and the anatomy reference below. "
        "There is no lookup table for this — reason from the location and the anatomy "
        "text each time.\n\n"
        + anatomy_text +
        "\n\nEach blob already has an N-class (N1=deep, N2=saphenous trunk, "
        "N3=superficial tributary) from a separate depth analysis — trust it, do not "
        "re-derive it. Your job is only to pick the specific vein name consistent with "
        "that N-class AND the probe location.\n\n"
        "STRICT VOCABULARY RULE: each blob below is listed with its OWN allowed "
        "vein_name choices, narrowed by its N-class. You MUST pick vein_name from "
        "EXACTLY that blob's own allowed list (or 'uncertain') — never a name from "
        "another blob's list, even if it seems to fit better. A saphenous trunk name "
        "like GSV/SSV can ONLY be used for an N2 blob; a deep-system name like CFV/FV/PV "
        "can ONLY be used for an N1 blob; 'Tributary' can ONLY be used for an N3 blob. "
        "If you believe a blob's own N-class must be wrong given what you see, you may "
        "still only choose from ITS allowed list or answer 'uncertain' — you cannot "
        "silently override its N-class here, that belongs to the earlier depth analysis, "
        "not this naming step.\n\n"
        "UNIQUENESS RULE: GSV, SSV, CFV, FV, and PV each name ONE specific real vessel — "
        "in a single frame's cross-section, the anatomy normally shows AT MOST ONE blob "
        "as GSV, AT MOST ONE as SSV, AT MOST ONE as CFV, etc. If two or more N2 blobs "
        "both look like plausible candidates for the same trunk name (e.g. both could be "
        "'GSV'), do NOT assign that name to more than one of them — decide which ONE "
        "blob is the actual main trunk (using position, size, and typical single-trunk "
        "anatomy: the true trunk is usually the more consistently-sized, centrally-"
        "located compartment vessel) and name the other blob(s) differently: as a "
        "genuinely different named vessel if the anatomy supports it (e.g. AASV/PASV/"
        "Giacomini running alongside), or reconsider whether that other blob might "
        "actually belong to a different N-class than currently assigned — but if it "
        "keeps its current N2 assignment and isn't a distinct named vessel, it should "
        "still receive a specific single-trunk name only once per frame; you are not "
        "required to force a second blob into a different name if the image genuinely "
        "doesn't support one — 'uncertain' is fine for the second blob rather than "
        "duplicating the trunk name.\n\n"
        "WHEN TO ANSWER 'uncertain' — DO NOT default to it just because the location "
        "confidence is medium/low or the leg_level says 'uncertain'. Confirmed real "
        "failure mode: a single, clean, unambiguous N2 blob got 'uncertain' for an "
        "entire clip instead of a real name — that is a failure to commit, not honest "
        "caution. You almost always have ENOUGH signal to make a well-reasoned specific "
        "call:\n"
        "- For an N2 blob specifically, GSV/SSV are the overwhelming anatomical default "
        "for this class (see 'GENERAL REASONING NOTES' above: front/medial surface or "
        "thigh/calf level -> GSV territory; posterior surface or popliteal fossa -> SSV "
        "territory) — even with only partial location evidence (e.g. surface known but "
        "leg_level uncertain, or vice versa), that is usually enough to commit to GSV or "
        "SSV rather than answering 'uncertain'. Only fall back to 'uncertain' for an N2 "
        "blob if you have NEITHER a usable surface NOR a usable leg_level to reason "
        "from at all.\n"
        "- For N1 and N3 blobs, use whatever partial location evidence is available the "
        "same way — a specific, well-reasoned answer graded by your own stated "
        "confidence in the reasoning field is more useful than a blank 'uncertain'.\n"
        "- Reserve 'uncertain' for when the image AND the location evidence together "
        "give you genuinely nothing to reason from (e.g. probe not visible at all, "
        "leg_side/leg_level/surface all uncertain simultaneously) — not merely when "
        "you're not fully sure.\n\n"
        "Respond with ONLY a compact JSON object, no markdown, no prose outside the "
        "JSON: "
        '{"<blob_id>": {"vein_name": "<from that blob\'s own allowed list, or '
        '\'uncertain\'>", "reasoning": "<one sentence>"}, ...}'
    )


def build_naming_prompt(blobs: list[dict], location: dict,
                         anatomy_text: str = anatomy_knowledge.ANATOMY_REFERENCE_TEXT) -> tuple[str, str]:
    """Pure function — testable with hand-built blob/location dicts, no image/model/network.
    blobs: [{"blob_id": int, "n_class": "N1"|"N2"|"N3", "centroid": [x, y]}, ...]
    location: Stage-3a output dict (see stage3_webcam_location.normalize)."""
    system = _system_prompt(anatomy_text)
    loc_desc = (
        f"Probe location (from webcam, confidence={location.get('confidence', 'uncertain')}): "
        f"leg_side={location.get('leg_side', 'uncertain')}, "
        f"leg_level={location.get('leg_level', 'uncertain')}, "
        f"surface={location.get('surface', 'uncertain')}. "
        f"Visual evidence: {location.get('visual_evidence', 'none')}"
    )
    blob_lines = [
        f"Blob {b['blob_id']}: n_class={b.get('n_class', '?')}, centroid={b.get('centroid')}, "
        f"allowed vein_name choices for THIS blob: {_allowed_names_for(b.get('n_class'))} or 'uncertain'"
        for b in blobs
    ]
    user = loc_desc + "\n\n" + "\n".join(blob_lines) + "\n\nName each blob."
    return system, user


MAX_NAMING_RETRIES = 3  # extra attempts beyond the first, asking again only for blobs
                         # that are missing, off-vocabulary, or in a duplicate-trunk-name
                         # conflict — see name_veins(). Raised from 2 -> 3: the final
                         # attempt now carries a forced tie-break instruction (see below)
                         # specifically so genuine duplicate conflicts get one extra
                         # real chance to resolve before falling back to "naming..." on
                         # both blobs in the rendered video.


def _name_veins_once(blobs: list[dict], location: dict, annotated_ultrasound_frame_bgr,
                      extra_instruction: str = "") -> dict:
    system, user = build_naming_prompt(blobs, location)
    if extra_instruction:
        user = user + "\n\n" + extra_instruction
    img_b64 = None
    if annotated_ultrasound_frame_bgr is not None:
        _, buf = cv2.imencode(".png", annotated_ultrasound_frame_bgr)
        img_b64 = base64.b64encode(buf).decode()
    parsed, _raw = groq_client.call_vlm_json(system, user, image_b64=img_b64, image_media_type="image/png",
                                              label="stage3b_naming")

    out = {}
    for key, val in parsed.items():
        try:
            bid = int(key)
        except ValueError:
            continue
        if isinstance(val, dict) and val.get("vein_name"):
            out[bid] = {"vein_name": val["vein_name"], "reasoning": val.get("reasoning", "")}
    return out


def _find_duplicate_trunk_names(named: dict) -> dict[str, list[int]]:
    """named: {blob_id: {"vein_name": str, ...}}. Returns {vein_name: [blob_id, ...]} for
    every UNIQUE_PER_FRAME_VEIN_NAMES entry claimed by 2+ blobs -- confirmed real bug this
    targets: two separate blobs both named "GSV" in one frame when only one real GSV
    exists. This is a structural self-consistency check on the model's OWN batch of
    answers (same category as vocabulary-membership validation below), not Python
    deciding which blob is actually correct -- see module docstring."""
    by_name: dict[str, list[int]] = {}
    for bid, val in named.items():
        name = val.get("vein_name")
        if name in anatomy_knowledge.UNIQUE_PER_FRAME_VEIN_NAMES:
            by_name.setdefault(name, []).append(bid)
    return {name: ids for name, ids in by_name.items() if len(ids) > 1}


def name_veins(blobs: list[dict], location: dict, annotated_ultrasound_frame_bgr=None) -> dict:
    """Returns {blob_id(int): {"vein_name": str, "reasoning": str}}.

    Every blob passed in gets a real, VLM-decided name if at all possible — the caller
    (pipeline.py) requires this so the final video never shows a bare N-class instead of
    a name, AND is expected to have already filtered out any blob with n_class=None (see
    pipeline.py) — this module has no vocabulary to offer a blob whose depth is unknown,
    so it should never be asked to name one.

    Three independent failure modes are checked after every attempt, each looping the
    affected blob_id(s) back for another try with a targeted corrective instruction
    (never a Python-decided replacement answer — always asks the model again):
      1. MISSING — model's JSON omitted a blob_id entirely (pre-existing check).
      2. OFF-VOCABULARY — model returned a vein_name outside that blob's own N-class-
         gated list (e.g. "Tributary" for an N2 blob) -- confirmed real bug #3.
      3. DUPLICATE TRUNK — model assigned the same unique-per-frame trunk name (GSV/SSV/
         CFV/FV/PV) to 2+ blobs in the same batch -- confirmed real bug #2.
    Blobs still unresolved after MAX_NAMING_RETRIES are left absent from the returned
    dict (same as the pre-existing missing-blob behavior) rather than Python forcing a
    guess — pipeline.py's hold logic retries again on the next naming tick, and the
    renderer falls back to showing the bare N-class rather than a name it can't trust.
    """
    remaining = {b["blob_id"]: b for b in blobs}
    by_id = {b["blob_id"]: b for b in blobs}
    out: dict = {}
    extra_instruction = ""
    attempt = 0
    while remaining and attempt <= MAX_NAMING_RETRIES:
        result = _name_veins_once(list(remaining.values()), location,
                                   annotated_ultrasound_frame_bgr, extra_instruction=extra_instruction)

        # --- Failure mode 2: OFF-VOCABULARY -- reject before ever merging into `out` ----
        problems: list[str] = []
        accepted: dict = {}
        for bid, val in result.items():
            blob = by_id.get(bid)
            if blob is None:
                continue
            allowed = _allowed_names_for(blob.get("n_class"))
            name = val["vein_name"]
            if name != "uncertain" and name not in allowed:
                problems.append(
                    f"Blob {bid} (n_class={blob.get('n_class')}): you answered '{name}', "
                    f"which is NOT in that blob's allowed list {allowed} — you must pick "
                    f"from ITS OWN list only, or 'uncertain'."
                )
                continue  # rejected -- stays in `remaining` for retry, never merged into out
            accepted[bid] = val

        out.update(accepted)
        for bid in list(remaining.keys()):
            if bid in accepted:
                del remaining[bid]

        # --- Failure mode 3: DUPLICATE TRUNK -- checked across the FULL accumulated
        # result (`out` now includes prior rounds too), since a conflict can span
        # attempts (e.g. blob 1 named GSV on attempt 0, blob 3 also named GSV on attempt
        # 1). Any conflicting blob is evicted from `out` and put back into `remaining`
        # regardless of which round it was accepted in.
        duplicates = _find_duplicate_trunk_names(out)
        for name, bids in duplicates.items():
            problems.append(
                f"You assigned the unique trunk name '{name}' to MORE THAN ONE blob "
                f"({bids}) in this same frame — only one real {name} can exist here. "
                f"Re-examine blobs {bids} and decide which ONE is the actual {name}; "
                f"rename the other(s) to a different specific vessel if the anatomy "
                f"supports it, or 'uncertain' if it doesn't."
            )
            for bid in bids:
                out.pop(bid, None)
                if bid in by_id:
                    remaining[bid] = by_id[bid]

        attempt += 1
        if remaining and attempt <= MAX_NAMING_RETRIES:
            missing_ids = sorted(remaining.keys())
            is_final_attempt = attempt == MAX_NAMING_RETRIES
            extra_instruction = (
                f"Blob(s) {missing_ids} still need a valid answer. "
                + (" ".join(problems) if problems else
                   "Your previous answer did not include a vein_name for these blob(s).")
                + f" You MUST provide a vein_name from EACH blob's own allowed list "
                  f"(shown above) for every one of {missing_ids} now, even if uncertain — "
                  f"do not omit any and do not reuse a trunk name already assigned to a "
                  f"different blob."
            )
            if is_final_attempt:
                # Confirmed real bug this guards against: without a forced tie-break, a
                # genuine duplicate-trunk conflict (e.g. two blobs both plausibly "GSV")
                # can survive every retry by the model just repeating the same conflicting
                # answer each time -- MAX_NAMING_RETRIES then exhausts with BOTH blobs
                # still unresolved, and the final video shows "naming..." on both instead
                # of a real answer on at least one. This is the LAST call for these
                # blob(s); force a decisive resolution rather than allowing another
                # repeat of the same unresolved conflict.
                extra_instruction += (
                    " THIS IS YOUR FINAL ATTEMPT for these blob(s) — if you are still "
                    "genuinely torn between two blobs for the same trunk name, you MUST "
                    "still resolve it now: pick whichever ONE blob you find even "
                    "slightly more consistent with that trunk vessel (position, size, "
                    "centrality in the compartment) and answer 'uncertain' for the "
                    "other(s), rather than repeating the same name on both again — a "
                    "repeated duplicate on this attempt will be discarded entirely and "
                    "BOTH blobs will end up unnamed, which is worse than one confident "
                    "answer plus one honest 'uncertain'."
                )
    return out
