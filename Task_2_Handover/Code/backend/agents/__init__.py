"""
Task 2 Agent Framework — CrewAI-backed streaming guidance pipeline.

Context-builder modules (plain Python, called before crew):
  history_agent  — session posY-band coverage + confirmed findings summary
  q_state_agent  — Q1-Q4 circuit status derived from confirmed clips
  protocol_agent — zone-specific examination protocol from medical literature
  vlm_agent      — ultrasound frame analysis (wrapper around vlm_analyzer)
  guidance_agent — state message builder + fallback helpers

CrewAI orchestration (5 agents, sequential context passing):
  crew_agents    — 5 Agent factories:
                     make_clinical_interpreter()  [ported from Task-1]
                     make_shunt_analyst()          [ported from Task-1]
                     make_circuit_analyst()        [Task-2 new]
                     make_navigation_planner()     [Task-2 new]
                     make_guidance_specialist()    [Task-2 new]
  crew_pipeline  — run_guidance_crew() chains all 5 and returns (guidance, raw, action)

Requires crewai (Python >=3.10,<=3.13). Use Task-1's Python 3.12 venv:
  C:\\Users\\Krish\\Downloads\\Cygnus_Med_Demo\\Task_1_App\\.venv\\Scripts\\python.exe
"""
from . import (
    history_agent,
    q_state_agent,
    protocol_agent,
    vlm_agent,
    guidance_agent,
    crew_agents,
    crew_pipeline,
)

__all__ = [
    "history_agent",
    "q_state_agent",
    "protocol_agent",
    "vlm_agent",
    "guidance_agent",
    "crew_agents",
    "crew_pipeline",
]
