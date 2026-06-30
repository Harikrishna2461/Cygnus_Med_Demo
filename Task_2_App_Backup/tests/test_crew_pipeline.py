"""
Direct test for the 5-agent CrewAI guidance pipeline.
Run with:  .venv\Scripts\python test_crew_pipeline.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from agents.crew_pipeline import run_guidance_crew
from agents import history_agent, q_state_agent, protocol_agent

# ── Fake session stub ─────────────────────────────────────────────────────────
class FakeSess:
    def __init__(self, clips):
        self.clips   = clips
        self.scan_log = [
            {"pos_y": 0.06, "region": "SFJ",     "surface": "anterior-medial", "leg": "right"},
            {"pos_y": 0.15, "region": "GSV-THI", "surface": "anterior-medial", "leg": "right"},
            {"pos_y": 0.30, "region": "GSV-THI", "surface": "medial",          "leg": "right"},
        ]


SCENARIOS = {
    "Q1_open": {
        "desc": "No clips yet — probe at SFJ zone",
        "clips": [],
        "region": "SFJ", "pos_y": 0.07, "surface": "anterior-medial",
        "leg": "right", "is_front": True,
    },
    "Q2_open": {
        "desc": "EP N1→N2 confirmed — Q2 open (does trunk reflux?)",
        "clips": [{"flow": "EP", "from_type": "N1", "to_type": "N2",
                   "pos_y_ratio": 0.06, "leg": "right", "elimination_test": ""}],
        "region": "GSV-THI", "pos_y": 0.20, "surface": "anterior-medial",
        "leg": "right", "is_front": True,
    },
    "Q3_open": {
        "desc": "EP N1→N2 + RP N2→N1 — Q3 open (where does trunk escape?)",
        "clips": [
            {"flow": "EP", "from_type": "N1", "to_type": "N2",
             "pos_y_ratio": 0.06, "leg": "right", "elimination_test": ""},
            {"flow": "RP", "from_type": "N2", "to_type": "N1",
             "pos_y_ratio": 0.25, "leg": "right", "elimination_test": ""},
        ],
        "region": "GSV-THI", "pos_y": 0.30, "surface": "medial",
        "leg": "right", "is_front": False,
    },
    "elim_test": {
        "desc": "Full trigger set — elimination test required",
        "clips": [
            {"flow": "EP", "from_type": "N1", "to_type": "N2",
             "pos_y_ratio": 0.06, "leg": "right", "elimination_test": ""},
            {"flow": "RP", "from_type": "N2", "to_type": "N1",
             "pos_y_ratio": 0.25, "leg": "right", "elimination_test": ""},
            {"flow": "EP", "from_type": "N2", "to_type": "N3",
             "pos_y_ratio": 0.28, "leg": "right", "elimination_test": ""},
            {"flow": "RP", "from_type": "N3", "to_type": "N1",
             "pos_y_ratio": 0.50, "leg": "right", "elimination_test": ""},
        ],
        "region": "GSV-CAL", "pos_y": 0.50, "surface": "posterior",
        "leg": "right", "is_front": False,
    },
}


def run_scenario(name: str, s: dict) -> None:
    sess  = FakeSess(s["clips"])
    hist  = history_agent.build_summary(sess, s["pos_y"])
    qst   = q_state_agent.analyze(s["clips"])
    proto = protocol_agent.get_protocol(s["region"], s["pos_y"])

    from agents.guidance_agent import build_state_message
    state_msg = build_state_message(
        region        = s["region"],
        pos_y         = s["pos_y"],
        surface       = s["surface"],
        leg           = s["leg"],
        clips         = s["clips"],
        vlm_summary   = "No frame provided.",
        history_summary = hist,
        q_state       = qst,
        protocol_text = proto,
        is_front      = s.get("is_front"),
    )

    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"DESC: {s['desc']}")
    print(f"{'='*70}")

    t0 = time.time()
    guidance, raw, action = run_guidance_crew(
        state_message   = state_msg,
        region          = s["region"],
        pos_y           = s["pos_y"],
        surface         = s["surface"],
        leg             = s["leg"],
        clips           = s["clips"],
        vlm_summary     = "No frame provided.",
        history_summary = hist,
        q_state         = qst,
        protocol_text   = proto,
        is_front        = s.get("is_front"),
    )
    elapsed = time.time() - t0

    print(f"ACTION  : {action}")
    print(f"GUIDANCE: {guidance}")
    print(f"TIME    : {elapsed:.1f}s")
    print(f"\nAGENT LOG (truncated):")
    for segment in raw.split(" | "):
        print(f"  {segment}")


if __name__ == "__main__":
    names = list(SCENARIOS.keys())
    if len(sys.argv) > 1:
        names = [sys.argv[1]]

    for name in names:
        run_scenario(name, SCENARIOS[name])
