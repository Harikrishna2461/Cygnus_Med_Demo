# Agentic System Architecture — CHIVA Task-2

## Overview
Every `probe_move` WebSocket event passes through a 4-stage agentic pipeline before a guidance instruction is emitted to the client.

---

## Stage 1 — Raw Inputs
| Input | Source |
|---|---|
| **Probe Position** | `region`, `pos_y`, `surface`, `leg`, `pos_x`, `is_front` from the WebSocket event |
| **Confirmed Clips** | `session.clips` — EP/RP marks the surgeon has confirmed so far |
| **Video Frame** | OpenCV extracts a JPEG at `pos_y × total_frames` from the annotated MP4 |
| **Scan Log** | `session.scan_log` — every probe position visited this session (max 300 entries) |

---

## Stage 2 — Context Agents (plain Python, no LLM)
Each agent reads one input and produces a structured text block.

| Agent | File | Output |
|---|---|---|
| **VLM Frame Agent** | `agents/vlm_agent.py` | Calls `vlm_analyzer.analyze_frame()` via Groq Vision API. Returns N1/N2/N3 vessel visibility and fascial layer status. Only runs if `|Δpos_y| ≥ 0.05`. |
| **Scan History Agent** | `agents/history_agent.py` | Counts visits per posY band (SFJ / upper thigh / Dodd / Hunterian / SPJ / calf / ankle). Reports visited vs unvisited bands. |
| **Q1–Q4 Circuit Agent** | `agents/q_state_agent.py` | Derives which of the 4 CHIVA diagnostic questions is currently open from the confirmed clip set. |
| **Protocol Agent** | `agents/protocol_agent.py` | Returns the zone-specific examination protocol (maneuvers, probe placement) for the current `region` + `pos_y`. Sources: Adler 2022, Gianesini 2014, Delfrate 2023, AVF 2023. |

All four outputs are merged by `guidance_agent.build_state_message()` into a single enriched prompt block containing: PROBE STATE · CONFIRMED FINDINGS · VLM FRAME · SCAN HISTORY · Q1–Q4 STATUS · PROTOCOL.

---

## Stage 3 — Deterministic Rule Gate
`_rule_based_action()` in `streaming_guidance_engine.py` is evaluated **before any LLM call**.

| Rule | Condition | Output |
|---|---|---|
| **maneuver** | EP N2→N3 + RP N3→N1 + RP N2→N1 all confirmed, no elimination test yet | `action="maneuver"` — compress tributary |
| **complete (Type 3)** | EP N2→N3 with `elimTest="No Reflux"` | `action="complete"` |
| **complete (Type 1+2)** | EP N2→N3 with `elimTest="Reflux"` | `action="complete"` |
| **complete (Type 4)** | EP N1→N3 + RP N2→N1 | `action="complete"` |
| **complete (Type 6)** | EP N1→N3 + RP N3→N1, no trunk involvement | `action="complete"` |
| **complete (Type 5)** | EP N1→N3 + RP N3→N2 + EP N2→N3 + RP N3→N1 | `action="complete"` |
| **complete (Type 1/2A)** | EP N1→N2 + RP N2→N1, no escape, max visited posY ≥ 0.48 | `action="complete"` |
| **complete (Type 2B)** | EP N2→N3 + RP N3→N1, no SFJ/trunk entry | `action="complete"` |
| **pass-through** | No rule fires | → proceed to Stage 4 |

If a rule fires, its output is emitted directly — the CrewAI crew is **not called**.

---

## Stage 4 — CrewAI 5-Agent Sequential Crew
`crew_pipeline.run_guidance_crew()` — only reached when the rule gate returns `(None, None)`.

**Framework:** CrewAI `Process.sequential` · ONE Crew · 5 Tasks · `context=[prev_task]` chains output downstream · LLM: `groq/llama-3.3-70b-versatile` via LiteLLM, temp=0.3.

| Agent | Role | Sees | Output limit |
|---|---|---|---|
| **Agent 1** | Clinical Interpreter | Full state message | ≤ 100 words |
| **Agent 2** | Shunt Analyst | Agent 1 output | ≤ 80 words |
| **Agent 3** | Circuit Analyst | Agent 1 + 2 output | ≤ 60 words |
| **Agent 4** | Navigation Planner | Agent 3 output | ≤ 50 words |
| **Agent 5** | Guidance Specialist | Agent 2 + 3 + 4 output | JSON only |

### Agent responsibilities
- **Clinical Interpreter** — Assesses whether confirmed clips are unambiguous, flags artefacts, checks VLM alignment.
- **Shunt Analyst** — Classifies the developing CHIVA shunt type (I, 2A, 2B, 2C, 3, 4, 5, 6) from the clip pattern.
- **Circuit Analyst** — Identifies which Q1–Q4 question is open and the exact anatomical zone (posY band) to examine next.
- **Navigation Planner** — Selects target posY band, probe surface, anatomical target, and maneuver (Paranà / Valsalva / squeeze).
- **Guidance Specialist** — Synthesises all upstream outputs into a single JSON probe-movement instruction ≤ 12 words.

### Safety guard
Even if the CrewAI crew returns `action="complete"` or `action="maneuver"`, the engine overrides it back to `action="move"`. Only the deterministic rule gate may authorise those actions.

---

## Output
```json
{"guidance": "Move distally along medial thigh toward Hunterian zone", "action": "move"}
{"guidance": "Compress tributary — record whether GSV Doppler changes",  "action": "maneuver"}
{"guidance": "Circuit mapped — sufficient findings for classification",   "action": "complete"}
```
Emitted to the client via SocketIO `guidance_update` event.

---

## External API
| Call | Model | Used by |
|---|---|---|
| Text LLM | `llama-3.3-70b-versatile` | All 5 CrewAI agents (temp=0.3) |
| Vision LLM | `meta-llama/llama-4-scout-17b-16e-instruct` | VLM Frame Agent (temp=0.0, max_tokens=250) |
| Fallback LLM | `llama-3.3-70b-versatile` | `guidance_agent.call_llm()` direct Groq SDK call (temp=0.0) |
