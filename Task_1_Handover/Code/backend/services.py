"""
Shared runtime services: Qdrant client, analysis cache, helpers.
Initialised once at startup and injected into route modules via module globals.
"""

import logging

from qdrant_client import QdrantClient

from config import QDRANT_PATH
from rag_engine import (
    retrieve_context,
    set_qdrant_client as set_rag_qdrant_client,
)
from general_chat_engine import (
    set_qdrant_client as set_general_qdrant_client,
)

logger = logging.getLogger(__name__)

# ── Singletons (populated by init_services) ───────────────────────────────────
qdrant_client: QdrantClient = None

# Per-session analysis context (in-memory, survives follow-up turns)
analysis_cache: dict[str, str] = {}


def init_services() -> None:
    """Create the shared Qdrant client; wire it into the engine modules."""
    global qdrant_client

    qdrant_client = QdrantClient(path=QDRANT_PATH)
    set_rag_qdrant_client(qdrant_client)
    set_general_qdrant_client(qdrant_client)

    logger.info("Services initialised (Qdrant).")


def retrieve_ligation_context(query: str, k: int = 5) -> list[str]:
    """Thin wrapper around rag_engine.retrieve_context() — the RAG entry point handed
    to the classification pipeline as retrieve_ligation_context_fn. Usage: passed by
    routes/clinical.py to both crew_pipeline.classify_and_plan_ligation_with_llm() (in
    /api/chat and /api/classify) as the function it calls to fetch ligation-planning
    passages for a given shunt type."""
    return retrieve_context(query, k=k)


def format_analysis_for_context(result: dict) -> str:
    """Serializes a classification result (shunt type, confidence, reasoning,
    ligation steps, rationale) into a short plain-text block. Usage: called by
    routes/clinical.py right after a successful classification, storing the result in
    services.analysis_cache[session_id] so build_conversational_response() has
    something to ground follow-up answers in during the same session."""
    lines: list[str] = []
    for f in result.get("findings", [result]):
        leg = f.get("leg", "Assessment")
        lines.append(
            f"LEG: {leg} | Shunt: {f.get('shunt_type','?')} "
            f"({f.get('confidence', 0):.0%} confidence)"
        )
        reasoning = f.get("reasoning", [])
        if reasoning:
            lines.append("Reasoning: " + " | ".join(reasoning[:3]))
        steps = f.get("ligation_steps", [])
        if steps:
            lines.append("Ligation steps: " + " -> ".join(steps[:3]))
        rationale = f.get("clinical_rationale", "")
        if rationale:
            lines.append(f"Rationale: {rationale[:300]}")
        lines.append("")
    return "\n".join(lines)
