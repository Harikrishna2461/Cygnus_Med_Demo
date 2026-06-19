"""
CLI test runner — replays a mock scenario against the live Task-2 API.

Usage:
    python tests/run_scenario.py [scenario_id] [--api http://localhost:7861]

    scenario_id: S1 | S2 | S3 | S4 | S5   (default: S2)
    --all        run all 5 scenarios sequentially
    --save       also save generated scenario JSON to tests/mock_scenarios/
    --no-vlm     skip sending the image (tests LLM path only)

The runner prints each event and the API guidance response side-by-side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

# Allow importing backend modules when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from mock_data_generator import generate_scenario, generate_all_scenarios, _SCENARIO_BUILDERS


def _short(text: str, n: int = 120) -> str:
    return text[:n] + "…" if len(text) > n else text


def run_scenario(scenario: dict, api_base: str, send_image: bool = True) -> list[dict]:
    """Send all events in a scenario to the API and collect responses."""
    print(f"\n{'='*70}")
    print(f"Scenario {scenario['id']}: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Leg: {scenario['leg']}  |  Expected shunt: {scenario['shunt_type'] or 'None'}")
    print(f"  Events: {len(scenario['events'])}")
    print(f"{'='*70}")

    results = []

    for ev in scenario["events"]:
        t = ev["timestamp"]
        etype = ev["type"]
        desc = ev["description"]
        sc = ev["scanner"]

        print(f"\n[t={t:5.1f}s] {etype.upper():20s} — {desc}")
        print(f"  Scanner: seg={sc['segment_id']} dist={sc['segment_dist']:.3f} "
              f"front={sc['is_front']}  x={sc.get('x')}  y={sc.get('y')}")

        # Build /api/guidance request
        body: dict = {
            "session_id": f"test_{scenario['id']}",
            "segment_id": sc["segment_id"],
            "segment_dist": sc["segment_dist"],
            "is_front": sc["is_front"],
            "x": sc.get("x"),
            "y": sc.get("y"),
            "expected_region": ev.get("expected_region"),
            "current_clip_index": 0,
        }

        if ev.get("ep_rp_findings"):
            body["ep_rp_findings"] = ev["ep_rp_findings"]

        if send_image and ev.get("image_b64"):
            body["ultrasound_image"] = ev["image_b64"]
            body["image_media_type"] = "image/jpeg"

        if ev.get("new_clip"):
            print(f"  + Clip marked: {ev['new_clip']['flow']} "
                  f"{ev['new_clip']['fromType']}→{ev['new_clip']['toType']}  "
                  f"posY={ev['new_clip']['posYRatio']:.3f}")

        try:
            resp = requests.post(f"{api_base}/api/guidance", json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            print(f"  ERROR: Cannot connect to {api_base}. Is the server running?")
            break
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"event": ev, "response": {"error": str(exc)}})
            continue

        # Print key parts of response
        loc = data.get("location", {})
        wr = data.get("wrong_region", {})
        print(f"  -> Region: {loc.get('region','?')} ({loc.get('leg','?')} leg, "
              f"conf={loc.get('confidence',0):.0%}, stable={loc.get('stable')})")

        if wr.get("is_wrong"):
            print(f"  !! WRONG REGION: {_short(wr.get('reason',''), 80)}")
            print(f"     Suggestion: {_short(wr.get('suggestion',''), 80)}")

        if data.get("current_guidance"):
            print(f"  => Guidance: {_short(data['current_guidance'])}")
        if data.get("next_step"):
            print(f"     Next:     {_short(data['next_step'])}")
        if data.get("clinical_note"):
            print(f"     Note:     {_short(data['clinical_note'])}")

        us = data.get("ultrasound_assessment")
        if us and not us.get("error"):
            print(f"  [VLM] quality={us.get('image_quality','?')}  "
                  f"mode={us.get('imaging_mode','?')}  "
                  f"target_visible={us.get('target_visible','?')}")
            if us.get("findings"):
                print(f"        findings: {_short(us['findings'], 100)}")

        results.append({"event": ev, "response": data})
        time.sleep(0.1)   # small throttle

    return results


def main():
    parser = argparse.ArgumentParser(description="Task-2 mock scenario test runner")
    parser.add_argument("scenario_id", nargs="?", default="S2",
                        choices=list(_SCENARIO_BUILDERS) + ["all"],
                        help="Scenario to run (default S2). Use 'all' for all.")
    parser.add_argument("--api", default="http://localhost:7861",
                        help="API base URL (default: http://localhost:7861)")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="Run all scenarios")
    parser.add_argument("--save", action="store_true",
                        help="Save generated scenario JSON files")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip image upload (test LLM-only path)")
    args = parser.parse_args()

    if args.save:
        from mock_data_generator import save_scenarios
        out = save_scenarios()
        print(f"Scenarios saved to: {out}")

    send_image = not args.no_vlm
    scenarios_to_run = (generate_all_scenarios()
                        if (args.run_all or args.scenario_id == "all")
                        else [generate_scenario(args.scenario_id)])

    all_results = {}
    for s in scenarios_to_run:
        results = run_scenario(s, args.api, send_image=send_image)
        all_results[s["id"]] = results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for sid, results in all_results.items():
        wrong = sum(1 for r in results if r["response"].get("wrong_region", {}).get("is_wrong"))
        errors = sum(1 for r in results if r["response"].get("error"))
        print(f"  {sid}: {len(results)} events  |  {wrong} wrong-region alerts  |  {errors} errors")


if __name__ == "__main__":
    main()
