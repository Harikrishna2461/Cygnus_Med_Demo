"""
Run all 38 CHIVA test cases against the live app (http://127.0.0.1:7860).
Prints a summary table of expected vs actual classifications.

Usage:
    python test_all_cases.py
"""

import json
import time
import requests

BASE_URL = "http://127.0.0.1:7860"

# ─────────────────────────────────────────────────────────────────────────────
# ALL 38 TEST CASES
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    # ── Type 1 ────────────────────────────────────────────────────────────────
    {
        "id": "T1-1", "expected": "Type 1",
        "question": "The SFJ is incompetent — blood enters the GSV from the deep system at the groin. The GSV then carries this blood backward down the thigh. There is no involvement of any tributary branches.",
    },
    {
        "id": "T1-2", "expected": "Type 1",
        "question": "At the saphenofemoral junction, deep venous blood refluxes into the great saphenous vein. The GSV is incompetent and blood travels retrograde along the entire thigh segment. No branches are filling from the GSV.",
    },
    {
        "id": "T1-3", "expected": "Type 1",
        "question": "Duplex shows SFJ incompetence with antegrade flow from femoral vein into GSV at the groin. GSV demonstrates retrograde flow throughout the thigh. No perforators or tributaries are involved.",
    },
    {
        "id": "T1-4", "expected": "Type 1",
        "question": "The deep-to-superficial junction at the groin is incompetent — blood flows forward from the deep system into the GSV. From there, the GSV refluxes backward without filling any side branches.",
    },
    {
        "id": "T1-5", "expected": "Type 1",
        "question": "A Hunterian perforator in the mid-thigh is incompetent, allowing deep venous blood to enter the GSV directly at that level. The GSV then carries blood backward both above and below. No tributaries are involved.",
    },
    # ── Type 2A ───────────────────────────────────────────────────────────────
    {
        "id": "T2A-1", "expected": "Type 2A",
        "question": "The SFJ is competent — no deep venous inflow into the GSV is detected. However the GSV is seen discharging blood forward into a tributary branch at the groin level, with no retrograde flow identified in the GSV or the tributary.",
    },
    {
        "id": "T2A-2", "expected": "Type 2A",
        "question": "Duplex scanning shows an isolated tributary filling from the GSV with forward flow. The SFJ is competent and closed. No reflux is present in the GSV trunk. The GSV feeds the branch but does not reflux.",
    },
    {
        "id": "T2A-3", "expected": "Type 2A",
        "question": "The GSV is mildly dilated at mid-thigh and discharges into a branch at that level. The SFJ shows no incompetence and there is no entry from the deep system. No reflux in any vessel — flow from GSV into the tributary is forward only.",
    },
    {
        "id": "T2A-4", "expected": "Type 2A",
        "question": "A superficial branch arises from the GSV at the knee level. Blood flows forward from the GSV into this branch. The SFJ is competent and no retrograde flow is detected anywhere in the limb.",
    },
    {
        "id": "T2A-5", "expected": "Type 2A",
        "question": "Blood is seen flowing forward from the GSV into a tributary near the knee region. The SFJ is competent, no perforator feeds the GSV from the deep system, and there is no retrograde flow in any vessel.",
    },
    # ── Type 2B ───────────────────────────────────────────────────────────────
    # True Type 2B: EP N2→N2 (perforator enters GSV) + RP N3→N1 or RP N3→N2, NO EP N1→N2, NO RP N2→N1
    {
        "id": "T2B-1", "expected": "Type 2B",
        "question": "The SFJ is competent. At the groin level a perforating vessel inserts into the GSV with forward antegrade flow. A tributary branch in the upper thigh then refluxes backward and drains into the deep venous system. The GSV trunk itself carries no retrograde flow.",
    },
    {
        "id": "T2B-2", "expected": "Type 2B",
        "question": "A communicating vessel at the proximal GSV enters the saphenous trunk with forward flow — the SFJ itself is competent and closed. A tributary in the mid-thigh shows retrograde flow, draining backward into the deep system. No backward flow in the GSV trunk.",
    },
    {
        "id": "T2B-3", "expected": "Type 2B",
        "question": "Duplex: SFJ is competent. A perforator feeds into the GSV near the groin with antegrade flow. At mid-thigh a varicose tributary demonstrates retrograde flow back toward the GSV. The main GSV does not reflux.",
    },
    {
        "id": "T2B-4", "expected": "Type 2B",
        "question": "The SFJ is competent. A perforating vessel at the upper thigh level inserts into the saphenous vein with forward flow. A tributary at the knee level carries blood backward — retrograde flow in the tributary draining toward the GSV. No retrograde flow in the GSV trunk.",
    },
    {
        "id": "T2B-5", "expected": "Type 2B",
        "question": "No SFJ incompetence is identified. A perforator inserts into the proximal GSV with antegrade forward flow. A branch at mid-thigh refluxes backward and drains into the deep vein via a perforating vessel. The GSV itself does not carry blood backward.",
    },
    # ── Type 2C ───────────────────────────────────────────────────────────────
    # True Type 2C: EP N2→N2 (perforator enters GSV) + RP N3→N2 + RP N2→N1, NO EP N1→N2
    {
        "id": "T2C-1", "expected": "Type 2C",
        "question": "The SFJ is competent. A perforating vessel at the groin inserts into the GSV with forward flow. A tributary at mid-thigh refluxes backward toward the GSV. Additionally the GSV trunk itself carries blood backward — secondary GSV reflux is present below the perforator entry. No SFJ incompetence.",
    },
    {
        "id": "T2C-2", "expected": "Type 2C",
        "question": "A communicating vessel enters the proximal GSV with antegrade flow — the SFJ is competent. Below the perforator entry the GSV carries blood backward. A tributary branch refluxes back toward the GSV as well. No deep-to-superficial incompetence at the SFJ.",
    },
    {
        "id": "T2C-3", "expected": "Type 2C",
        "question": "SFJ competent. A perforator at the groin level feeds the GSV forward. A tributary at the calf refluxes backward into the deep system via a perforating vessel. The GSV trunk between the perforator entry and the tributary shows secondary retrograde flow. No SFJ incompetence.",
    },
    {
        "id": "T2C-4", "expected": "Type 2C",
        "question": "Duplex: SFJ closed and competent. A small perforating vessel inserts into the proximal GSV. Below that entry point the GSV refluxes backward. A tributary at the knee carries retrograde flow back toward the GSV. No SFJ entry point — the SFJ is not incompetent.",
    },
    {
        "id": "T2C-5", "expected": "Type 2C",
        "question": "The SFJ is competent. A perforator at the upper thigh inserts into the GSV with forward antegrade flow. The GSV below the perforator carries blood backward. A branch at mid-calf refluxes backward toward the GSV. No incompetence at the SFJ.",
    },
    # ── Type 3 ────────────────────────────────────────────────────────────────
    {
        "id": "T3-1", "expected": "Type 3",
        "question": "The SFJ is incompetent — deep blood enters the GSV at the groin. The GSV refluxes down to the knee. At the knee the GSV feeds a tributary — blood escapes from the GSV into a branch at the knee level. That tributary then carries blood backward toward the GSV — retrograde flow back toward the saphenous trunk. The GSV does not reflux beyond the knee.",
    },
    {
        "id": "T3-2", "expected": "Type 3",
        "question": "There is SFJ incompetence with antegrade flow from the deep system into the GSV. The GSV carries reflux down to mid-thigh. At mid-thigh the GSV feeds a branch with forward flow, and that branch shows retrograde flow. No reflux in the main GSV beyond the branch.",
    },
    {
        "id": "T3-3", "expected": "Type 3",
        "question": "Blood enters the GSV from the deep system at the groin (SFJ incompetent). The GSV refluxes down to the calf. In the calf the GSV discharges blood forward into a branch, and no retrograde flow in the GSV below that point. The branch carries retrograde flow back toward the GSV.",
    },
    {
        "id": "T3-4", "expected": "Type 3",
        "question": "Duplex shows SFJ incompetence. Blood flows antegrade from deep into GSV at the groin, then the GSV refluxes to the ankle. At the ankle it feeds a superficial branch with forward flow. The branch refluxes backward toward the GSV — retrograde flow from the tributary back toward the saphenous trunk. The GSV itself does not reflux below the tributary junction.",
    },
    {
        "id": "T3-5", "expected": "Type 3",
        "question": "A Hunterian perforator in the upper thigh is incompetent and blood enters the GSV from the deep system at that level. The GSV refluxes both above and below the entry point. At the knee the GSV discharges forward into a tributary. That tributary carries blood backward toward the GSV — retrograde flow back toward the saphenous trunk. No SFJ incompetence.",
    },
    # ── Type 1+2 ──────────────────────────────────────────────────────────────
    {
        "id": "T12-1", "expected": "Type 1+2",
        "question": "The SFJ is incompetent — blood enters the GSV at the groin from the deep system. The GSV refluxes throughout the thigh. At mid-thigh the GSV feeds a tributary, and that tributary drains backward into the deep venous system via a perforating vessel. When the mid-thigh GSV was compressed, the tributary reflux disappeared, confirming the GSV is the feeding source.",
    },
    {
        "id": "T12-2", "expected": "Type 1+2",
        "question": "SFJ incompetent with full-length GSV reflux. A tributary at the knee fills from the GSV and drains backward into the deep system via a perforator — the tributary re-enters the deep venous system. Elimination test performed at the knee — tributary reflux abolished on compression, confirming Type 1+2 combined shunt.",
    },
    {
        "id": "T12-3", "expected": "Type 1+2",
        "question": "Blood enters the GSV from the deep system at the groin. The entire GSV refluxes. A large tributary at the calf arises from the GSV and connects back to the deep vein through a perforating vessel — the tributary drains to deep via perforating vessel. Elimination test at the GSV feeding point eliminated all tributary reflux.",
    },
    {
        "id": "T12-4", "expected": "Type 1+2",
        "question": "SFJ incompetence confirmed. GSV refluxes from groin to calf. At the calf, a tributary branch fills from the GSV and re-enters the deep system via a perforator. Compression of the GSV above the calf tributary abolished all tributary reflux — elimination test positive.",
    },
    {
        "id": "T12-5", "expected": "Type 1+2",
        "question": "Full-length GSV reflux with SFJ incompetence. A large tributary at mid-thigh fills from the GSV and drains backward into the deep venous system via a perforating vessel. A Trendelenburg test (compression of the groin) eliminated all reflux in both the GSV and the tributary, confirming single source shunt with combined Type 1+2 pattern.",
    },
    # ── No Shunt ──────────────────────────────────────────────────────────────
    {
        "id": "NS-1", "expected": "No Shunt",
        "question": "The SFJ is competent and shows no deep-to-superficial incompetence. The GSV demonstrates forward antegrade flow throughout the thigh. No retrograde flow is detected anywhere in the GSV or any tributary branches. The deep venous system shows normal phasic flow.",
    },
    {
        "id": "NS-2", "expected": "No Shunt",
        "question": "Duplex assessment shows normal venous function. The SFJ is competent with no reflux. The GSV demonstrates physiological forward flow only. No retrograde flow is identified anywhere. No tributary varicosities are seen.",
    },
    {
        "id": "NS-3", "expected": "No Shunt",
        "question": "Complete duplex assessment reveals a competent venous system. The SFJ is competent with no deep-to-superficial incompetence. The GSV is patent with physiological forward flow throughout. No retrograde flow detected anywhere in the limb. The system is completely reflux-free.",
    },
    {
        "id": "NS-4", "expected": "No Shunt",
        "question": "Venous duplex: The SFJ is competent. There is no evidence of deep-to-superficial incompetence. The GSV is patent with antegrade flow. No reflux is present anywhere — no retrograde flow in the GSV trunk, no tributary reflux, and no perforator incompetence detected.",
    },
    # ── Additional No Shunt cases ─────────────────────────────────────────────
    {
        "id": "NS-5", "expected": "No Shunt",
        "question": "The SFJ is competent. The GSV carries blood forward in physiological direction throughout. No retrograde flow is detected anywhere in the GSV or any tributary branches. No perforator incompetence is present. No pathological shunt identified.",
    },
    {
        "id": "NS-6", "expected": "No Shunt",
        "question": "Duplex shows normal venous hemodynamics. SFJ is closed and competent. The GSV shows forward antegrade flow from groin to calf. No retrograde flow identified anywhere in the system — no GSV reflux, no tributary filling, no perforator incompetence.",
    },
    {
        "id": "NS-7", "expected": "No Shunt",
        "question": "The deep venous system is normal with phasic flow. The SFJ is competent and no deep-to-superficial reflux is present. The entire GSV trunk shows forward physiological flow. No retrograde flow is detected anywhere. Patient has no pathological shunt.",
    },
    {
        "id": "NS-8", "expected": "No Shunt",
        "question": "Venous assessment shows a competent venous system throughout. The GSV is patent and non-refluxing. No retrograde flow is detected anywhere in the limb — not at the SFJ, not in the GSV trunk, and no tributary filling. Venous function is normal.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────

def create_session():
    r = requests.post(f"{BASE_URL}/api/session", json={"title": "Automated Test", "mode": "clinical"}, timeout=10)
    r.raise_for_status()
    return r.json()["session_id"]


def run_case(session_id: str, question: str) -> dict:
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={"session_id": session_id, "message": question},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def extract_shunt_type(response: dict) -> str:
    # Try top-level shunt_type first
    st = response.get("shunt_type") or ""
    if st:
        return st
    # Try findings list
    findings = response.get("findings", [])
    if findings:
        return findings[0].get("shunt_type") or ""
    return "ERROR: no shunt_type"


def normalize(shunt_type: str) -> str:
    """Normalise to expected label for comparison."""
    s = shunt_type.lower().strip()
    if "no shunt" in s:
        return "No Shunt"
    if "undetermined" in s:
        return "Undetermined"
    if "1+2" in s or "1 + 2" in s or "combined" in s:
        return "Type 1+2"
    if "type 1" in s:
        return "Type 1"
    if "2a" in s:
        return "Type 2A"
    if "2b" in s:
        return "Type 2B"
    if "2c" in s:
        return "Type 2C"
    if "type 3" in s or "type3" in s:
        return "Type 3"
    return shunt_type  # return raw if nothing matched


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("CHIVA 38-Case Automated Test")
    print("=" * 72)

    # Check server alive
    try:
        requests.get(f"{BASE_URL}/api/status", timeout=5)
    except Exception:
        print("\n[ERROR] Cannot reach server at", BASE_URL)
        print("Start the app first:  cd backend && python app.py")
        return

    results = []
    passed = 0
    failed = 0

    for tc in TEST_CASES:
        sid = create_session()
        try:
            resp = run_case(sid, tc["question"])
        except Exception as e:
            actual = f"HTTP ERROR: {e}"
            ok = False
        else:
            actual_raw = extract_shunt_type(resp)
            actual = normalize(actual_raw)
            ok = normalize(tc["expected"]) == actual

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {tc['id']:8s}  expected={tc['expected']:10s}  got={actual}")
        results.append({**tc, "actual": actual, "ok": ok})

        # Small delay to avoid rate-limiting on Groq
        time.sleep(1.5)

    print()
    print("=" * 72)
    print(f"RESULTS:  {passed}/{len(TEST_CASES)} passed  ({100*passed//len(TEST_CASES)}%)")
    print("=" * 72)

    if failed:
        print("\nFAILED CASES:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['id']:8s}  expected={r['expected']:10s}  got={r['actual']}")

    # Save JSON report
    import datetime, pathlib
    out = pathlib.Path(__file__).parent / f"test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved to: {out.name}")


def run_clip_file_tests():
    """
    Test suite 2: send actual clip JSON files directly to the classifier,
    bypassing the NL interpreter entirely.  Reads every clipdata*.json from
    sample_data/data/<ShuntType>/ and verifies the
    classifier returns the expected shunt type.
    """
    import os, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "backend"))

    try:
        from config import GROQ_API_KEY, GROQ_MODEL
        from shunt_classification_and_ligation_llm import classify_and_plan_ligation_with_llm
        import requests as req
    except Exception as e:
        print(f"[SKIP] Clip-file tests: could not import backend ({e})")
        return

    def _call_llm(prompt, return_usage=False):
        r = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
            timeout=60,
        )
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        ud = {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}
        return (text, ud) if return_usage else text

    data_root = pathlib.Path(__file__).parent / "sample_data" / "data"
    if not data_root.exists():
        print("[SKIP] Clip-file tests: sample_data/data not found")
        return

    # Map folder names to expected normalised shunt type
    FOLDER_EXPECTED = {
        "Shunt_Type_1":   "Type 1",
        "Shunt_Type_1+2": "Type 1+2",
        "Shunt_Type_2A":  "Type 2A",
        "Shunt_Type_2B":  "Type 2B",
        "Shunt_Type_2C":  "Type 2C",
        "Shunt_Type_3":   "Type 3",
        "No_Shunt":       "No Shunt",
    }

    print()
    print("=" * 72)
    print("CHIVA Clip-File Direct Tests (no NL interpreter)")
    print("=" * 72)

    passed = failed = 0
    for folder, expected in FOLDER_EXPECTED.items():
        folder_path = data_root / folder
        if not folder_path.exists():
            continue
        clip_files = sorted(folder_path.glob("clipdata*.json"))
        for clip_file in clip_files:
            with open(clip_file) as f:
                raw = json.load(f)
            clips = raw.get("clips", [])
            if not clips:
                continue
            label = f"{folder}/{clip_file.name}"
            try:
                result = classify_and_plan_ligation_with_llm(clip_list=clips, call_llm_fn=_call_llm)
                got_raw = result.get("shunt_type", "ERROR")
                got = normalize(got_raw)
            except Exception as e:
                got = f"ERROR: {e}"
            ok = normalize(expected) == got
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"[{status}] {label:40s}  expected={expected:10s}  got={got}")
            time.sleep(1.0)

    total = passed + failed
    print()
    print("=" * 72)
    if total:
        print(f"CLIP-FILE RESULTS:  {passed}/{total} passed  ({100*passed//total}%)")
    else:
        print("CLIP-FILE RESULTS:  no clip files found")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    if "--clips" in sys.argv:
        run_clip_file_tests()
    elif "--all" in sys.argv:
        main()
        run_clip_file_tests()
    else:
        main()
