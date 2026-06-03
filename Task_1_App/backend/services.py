"""
Shared runtime services: LLM client, Qdrant client, analysis cache, helpers.
Initialised once at startup and injected into route modules via module globals.
"""

import logging
from pathlib import Path

from groq import Groq as GroqClient
from qdrant_client import QdrantClient

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    QDRANT_PATH, QDRANT_COLLECTION,
    RERANK_TOP_N,
)
from rag_engine import (
    retrieve_context,
    set_qdrant_client as set_rag_qdrant_client,
)
from general_chat_engine import (
    set_qdrant_client as set_general_qdrant_client,
)

logger = logging.getLogger(__name__)

# ── Singletons (populated by init_services) ───────────────────────────────────
groq_client: GroqClient = None
qdrant_client: QdrantClient = None

# Per-session analysis context (in-memory, survives follow-up turns)
analysis_cache: dict[str, str] = {}


def init_services() -> None:
    """Create shared Groq and Qdrant clients; wire into engine modules."""
    global groq_client, qdrant_client

    groq_client = GroqClient(api_key=GROQ_API_KEY)

    qdrant_client = QdrantClient(path=QDRANT_PATH)
    set_rag_qdrant_client(qdrant_client)
    set_general_qdrant_client(qdrant_client)

    logger.info("Services initialised (Groq + Qdrant).")


def call_llm(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1536,
    return_usage: bool = False,
):
    """Call the Groq LLM. Returns (text, usage) when return_usage=True."""
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        if return_usage:
            u = resp.usage
            usage = {
                "prompt_tokens":     getattr(u, "prompt_tokens",     0) if u else 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0,
                "total_tokens":      getattr(u, "total_tokens",      0) if u else 0,
            }
            return text, usage
        return text
    except Exception as e:
        logger.error(f"Groq LLM error: {e}")
        err = f"LLM error: {e}"
        if return_usage:
            return err, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return err


def retrieve_ligation_context(query: str, k: int = 5) -> list[str]:
    return retrieve_context(query, k=k)


def format_analysis_for_context(result: dict) -> str:
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
