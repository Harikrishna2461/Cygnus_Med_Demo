"""
CrewAI tool definitions for the Task-2 guidance pipeline.

These tools expose the deterministic protocol lookup functions so that
agents — primarily NavigationPlanner — can query the protocol for ANY
zone, not just the zone pre-selected at request time.

Core quality problem this solves:
  The protocol_agent pre-selects a protocol based on the CURRENT probe
  position before the crew runs. CircuitAnalyst then decides to route
  the probe to a DIFFERENT zone, but NavigationPlanner still has the
  wrong zone's protocol in its context.

  With get_zone_protocol(), NavigationPlanner calls the tool with the
  TARGET zone and gets the correct examination rules before composing
  its movement command.

Stateless tools (module-level, thread-safe):
  get_zone_protocol             — protocol for any named zone
  get_vein_examination_objectives — Mendoza 2014 objectives for any vein

Stateful tools (built as closures inside run_guidance_crew):
  Not defined here — see crew_pipeline.make_stateful_tools().
"""
from __future__ import annotations

from crewai.tools import tool

# ── Zone key normalisation map ─────────────────────────────────────────────────
# Maps any reasonable name an LLM might use → _PROTOCOLS dict key
_ZONE_KEY_MAP: dict[str, str] = {
    # SFJ / groin
    "sfj":                 "sfj_groin",
    "sfj_groin":           "sfj_groin",
    "groin":               "sfj_groin",
    "sfj/groin":           "sfj_groin",
    "sfj groin":           "sfj_groin",
    # Upper thigh
    "upper_thigh":         "upper_thigh",
    "upper thigh":         "upper_thigh",
    "upperthigh":          "upper_thigh",
    # Hunterian
    "hunterian":           "hunterian_proximal",
    "hunterian_proximal":  "hunterian_proximal",
    "proximal thigh":      "hunterian_proximal",
    "proximal_thigh":      "hunterian_proximal",
    "hunterian proximal":  "hunterian_proximal",
    "hunter":              "hunterian_proximal",
    # Dodd
    "dodd":                "dodd_distal",
    "dodd_distal":         "dodd_distal",
    "distal thigh":        "dodd_distal",
    "distal_thigh":        "dodd_distal",
    "dodd distal":         "dodd_distal",
    # Popliteal / SPJ
    "popliteal":           "popliteal_spj",
    "spj":                 "popliteal_spj",
    "popliteal_spj":       "popliteal_spj",
    "popliteal spj":       "popliteal_spj",
    "popliteal/spj":       "popliteal_spj",
    # Calf
    "calf":                "calf",
    # Ankle / SSV
    "ankle":               "ankle_ssv",
    "ankle_ssv":           "ankle_ssv",
    "ssv":                 "ankle_ssv",
    "ankle ssv":           "ankle_ssv",
    # General sequence
    "general":             "general_sequence",
    "general_sequence":    "general_sequence",
    "sequence":            "general_sequence",
}


@tool("get_zone_protocol")
def get_zone_protocol(zone: str) -> str:
    """
    Returns the book-sourced duplex ultrasound examination protocol for a named
    anatomical zone. Call this with the TARGET zone when directing the probe to
    a zone different from the current position — this ensures the navigation
    command is grounded in the correct protocol rules for that zone.

    Accepted zone names (case-insensitive):
      sfj / sfj_groin / groin
      upper_thigh / upper thigh
      hunterian / hunterian_proximal / proximal thigh
      dodd / dodd_distal / distal thigh
      popliteal / spj / popliteal_spj
      calf
      ankle / ankle_ssv / ssv
      general / general_sequence
    """
    from agents import protocol_agent  # lazy import avoids circular deps at load time

    key = _ZONE_KEY_MAP.get(zone.lower().strip().replace("-", "_"))
    if key:
        return protocol_agent._PROTOCOLS[key]
    # Fallback: try direct key lookup in case the LLM used an exact key
    direct = protocol_agent._PROTOCOLS.get(zone.lower().strip())
    if direct:
        return direct
    return (
        f"Zone '{zone}' not recognised. Valid names: "
        "sfj_groin, upper_thigh, hunterian_proximal, dodd_distal, "
        "popliteal_spj, calf, ankle_ssv, general_sequence"
    )


@tool("get_vein_examination_objectives")
def get_vein_examination_objectives(vein: str) -> str:
    """
    Returns the comprehensive Mendoza 2014 examination objectives for a specific
    vein. Use when the operator is in vein scan mode and the full checklist of
    clinical questions must inform the navigation command.

    Valid vein values (case-insensitive):
      GSV, SSV, PERFORATORS, TRIBUTARIES, DEEP_VEINS
    """
    from agents import protocol_agent

    result = protocol_agent.get_vein_examination_protocol(vein.upper().strip())
    if result:
        return result
    return (
        f"Vein '{vein}' not recognised. "
        "Valid values: GSV, SSV, PERFORATORS, TRIBUTARIES, DEEP_VEINS"
    )


# ── Public tool lists ──────────────────────────────────────────────────────────
# Import these lists in crew_agents.py to assign tools to agents.

# NavigationPlanner tools removed — llama-3.3-70b generates malformed tool call JSON
# that Groq's native tool-calling rejects, crashing the entire crew.
# Routing instructions come from CircuitAnalyst context instead.
NAVIGATION_PLANNER_TOOLS: list = []

# CircuitAnalyst reasons purely from clips + scan history; it has no task instruction
# to call any tool, so its tool list is empty.
CIRCUIT_ANALYST_TOOLS: list = []
