# Duplex Ultrasound Scanning Protocols — Reference Document

Extracted from PDF review (June 2026) for Task 2 active guidance system.
Integrated into `backend/agents/protocol_agent.py` and `backend/agents/guidance_agent.py`.

---

## Sources

| Reference | File in books_articles/ |
|-----------|------------------------|
| Adler et al. 2022 | `adler-et-al-2022-varicose-veins-of-the-lower-extremity-doppler-us-evaluation-protocols-patterns-and-pitfalls.pdf` |
| Gianesini et al. 2014 | `CHIVA STRATEGY Gianesini014.pdf` |
| Delfrate 2023 | `DelfrateR CHIVA article.pdf` |
| AVF 2023 guidelines | Referenced in Delfrate 2023 |
| **Mendoza et al. 2014** | `0-duplex-ultrasound-of-superficial-leg-veins-2014.pdf` — Sections 7.2 (GSV), 8.2 (SSV), 9.2 (Perforators), 10.2 (Tributaries), Ch. 14 (Deep veins) |

---

## 1. Patient Positioning

**Source: Adler et al. 2022, p. 2190**

- **Reverse Trendelenburg ≥60°** for ALL venous insufficiency studies — legs must be below the level of the patient's head to maximise venous filling and optimise reflux detection.
- Hip externally rotated, knee slightly flexed (non-weight bearing).
- **Left lateral decubitus** for RIGHT SSV assessment.
- **Right lateral decubitus** for LEFT SSV assessment.

---

## 2. Valsalva vs Paranà vs Squeezing

**Source: Adler et al. 2022 p. 2190–2191; Delfrate 2023 p. 21; Gianesini 2014 p. 12**

| Maneuver | Technique | When to use |
|----------|-----------|-------------|
| **Valsalva** | Blocked forced expiration; adequate when forward CFV flow ceases | SFJ, SPJ, upper deep veins |
| **Paranà** | Slight waist push triggers calf proprioceptive contraction-relaxation reflex | Preferred for all reflux testing; more physiological than squeezing |
| **Squeezing** | Manual calf squeeze and release | Acceptable but insufficient alone; use Paranà as primary |

**Critical (Delfrate 2023):** "Squeezing alone in a patient standing or sitting is not sufficient, and creates a risk of overtreatment." Always confirm with Paranà or Valsalva.

---

## 3. SFJ Assessment Protocol

**Source: Gianesini 2014 p. 12; Delfrate 2023 pp. 22–23**

1. Transverse B-mode: identify **"Mickey Mouse" sign** — CFV in centre, GSV and femoral artery as lateral ovals.
2. Place Doppler sample gate on the **FEMORAL SIDE** of the terminal valve.
3. Apply **Valsalva**: confirm adequacy by cessation of forward CFV flow.
4. Apply **Paranà/squeeze** (calf proprioceptive maneuver).
5. **BOTH Valsalva AND Paranà must be positive** to confirm SFJ incompetence (N1→N2 entry).
6. If only Paranà positive but not Valsalva → terminal valve competent; reflux is pre-terminal or from a **pelvic leak point (PLP)**.
7. Three pelvic leak points to check: Superior Gluteal Point (SGP), Inferior Gluteal Point (IGP), Obturator Point (OP).
8. **AASV** (anterior accessory saphenous vein) lies anterior to GSV in upper thigh — assess separately; classified **N3, not N2** (common duplex pitfall).

---

## 4. SPJ Assessment Protocol

**Source: Gianesini 2014 p. 12; Delfrate 2023 p. 23**

1. **Both Paranà (active) AND compression/relaxation (passive CR) must be positive simultaneously** to confirm SPJ incompetence.
2. One maneuver positive alone ≠ true junctional incompetence.
3. SPJ location is **variable** — may connect to gastrocnemian vein rather than popliteal vein directly.
4. **Giacomini vein** (posterior thigh, SSV→GSV): forward systolic flow with Paranà = viable outflow route.
5. SPJ disconnection should be performed below the Giacomini junction in mixed shunts.

---

## 5. GSV/SSV Trunk Assessment Protocol

**Source: Adler et al. 2022 pp. 2191; Delfrate 2023 p. 24**

1. **Transverse B-mode ("saphenous eye")**: GSV should sit within the fascial compartment between two bright horizontal fascia lines.
2. Confirm N2 identity by the fascial envelope — "superficialization" (escape from compartment) changes treatment options.
3. Measure anteroposterior **GSV diameter** at upper thigh, mid-thigh, above knee, below knee.
4. **Reflux threshold**: outward/reversed flow lasting **>500 ms** on Paranà = haemodynamically significant.
   - Some guidelines use 1 second for better specificity; 500 ms is the conventional threshold.
5. "No reflux no re-entry": if GSV reflux persists below an escape point → another re-entry exists further distal.

---

## 6. Perforator Assessment Criteria

**Source: Adler et al. 2022 pp. 2186, 2191–2192; Delfrate 2023 pp. 23–25; AVF 2023**

**Three maneuvers required (Delfrate 2023):**
1. Static squeezing (gravitational test)
2. Paranà maneuver (physiological proprioceptive — preferred)
3. Valsalva (hypertensive test)

**Pathological perforator (AVF 2023 guidelines):** outward flow **≥500 ms** AND diameter **≥3.5 mm**.

**Re-entry identification:**
- Inward flow during muscle **diastole** (Paranà release phase) = re-entry perforator (RP N3→N1).
- Diastolic reflux into deep system via perforator is **always pathological and pathogenic**.
- **Biphasic perforators**: systolic outward + diastolic inward flow = likely re-entry candidate (hemodynamically, inward diastolic phase dominates).

**Systolic outward vs diastolic inward:**
- Systolic outward flow = may or may not be pathogenic (depends on return path).
- Diastolic inward flow = always re-entry; hemodynamically significant.

---

## 7. Standard Examination Sequence

**Source: Adler et al. 2022 p. 2190; Delfrate 2023 p. 21**

1. **DVT compression** — all deep veins, before any reflux testing.
2. **Deep vein reflux** — iliac valve competence (Valsalva), CFV above SFJ.
3. **SFJ** — Mickey Mouse sign; Valsalva + Paranà (both required); AASV separate.
4. **GSV trunk** — medial thigh → calf; saphenous eye; 500 ms reflux threshold.
5. **Hunterian perforators** — all 3 maneuvers; check for N3 above fascia (EP N2→N3).
6. **SPJ** — posterior knee; both Paranà + CR required; variable anatomy.
7. **SSV trunk** — posterior calf; lateral approach; same 500 ms threshold.
8. **Re-entry perforators** — inward diastolic flow; confirm ≥3.5 mm; biphasic pattern.

---

## 8. Where This Information Is Used in the Code

| Protocol element | Code location |
|-----------------|--------------|
| Zone-specific protocol text | `backend/agents/protocol_agent.py` → `_PROTOCOLS` dict |
| System prompt protocol section | `backend/agents/guidance_agent.py` → `SYSTEM_PROMPT` (REGION-SPECIFIC PROTOCOL KNOWLEDGE) |
| Perforator criteria (500 ms, 3.5 mm) | Mentioned in both protocol_agent and guidance_agent system prompt |
| Paranà vs squeezing clarification | guidance_agent SYSTEM_PROMPT |
| Saphenous eye transverse confirmation | guidance_agent SYSTEM_PROMPT |
| Patient positioning (Reverse Trendelenburg) | guidance_agent SYSTEM_PROMPT |
| **Vein-specific exam objectives (Mendoza 2014)** | `backend/agents/protocol_agent.py` → `_VEIN_EXAM_OBJECTIVES` dict + `get_vein_examination_protocol()` |
| **Vein scan mode state** | `backend/streaming_session.py` → `StreamSession.scan_vein` field |
| **Vein scan mode socket event** | `backend/routes/stream.py` → `set_scan_vein` / `scan_vein_ack` events |

### Vein Mode — How It Works

When the operator emits `set_scan_vein` with `{"session_id": "...", "vein": "GSV"}`,
`StreamSession.scan_vein` is set to `"GSV"`. On every subsequent `probe_move`, the
guidance engine calls `protocol_agent.get_protocol(region, pos_y, vein_mode="GSV")`,
which appends the full GSV examination checklist (27 questions from Ch. 7.2) to the
zone-specific protocol. The LLM then navigates the probe within the zone context AND
reminds the examiner which clinical questions still need answering.

Valid `vein` values: `"GSV"`, `"SSV"`, `"PERFORATORS"`, `"TRIBUTARIES"`, `"DEEP_VEINS"`, `""` (clear mode).
