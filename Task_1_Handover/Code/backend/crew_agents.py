"""
CrewAI agent definitions for the CHIVA clinical pipeline.

Three agents cover all LLM calls in the system:
  make_clinical_interpreter()      - sufficiency check, CHIVA clip generation, follow-up Q&A
  make_shunt_analyst()             - shunt type classification, CHIVA ligation planning
  make_general_medical_assistant() - general medical Q&A with RAG-provided context

Each factory creates a fresh Agent instance; do not share instances across requests.
"""

import logging

import litellm
from crewai import Agent, LLM

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# crewai's internal flow system injects 'cache_breakpoint' into message dicts
# before calling litellm. Groq rejects any message with unknown properties.
# Patch litellm.completion to strip it before the HTTP request is made.
_orig_completion = litellm.completion


def _patched_completion(*args, **kwargs):
    for msg in kwargs.get("messages", []):
        msg.pop("cache_breakpoint", None)
    return _orig_completion(*args, **kwargs)


litellm.completion = _patched_completion


def _make_llm() -> LLM:
    return LLM(
        model=f"groq/{GROQ_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )


def make_clinical_interpreter() -> Agent:
    """
    Covers nl_interpreter.py LLM calls:
      - Sufficiency check (_SUFFICIENCY_PROMPT)
      - CHIVA clip generation (_NL_TO_CHIVA_PROMPT)
      - Conversational follow-up (_CONVERSATIONAL_PROMPT)
    """
    return Agent(
        role="CHIVA Clinical Interpreter",
        goal=(
            "Assess whether a surgeon's clinical description contains sufficient flow information, "
            "translate it into CHIVA virtual clip notation, and answer post-classification questions."
        ),
        backstory=(
            "Expert CHIVA vascular surgeon with 20 years of duplex scanning experience. "
            "Fluent in EP/RP, N1/N2/N3, SFJ, SPJ, GSV, SSV notation. "
            "Never fabricates flow events not explicitly stated by the clinician."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_shunt_analyst() -> Agent:
    """
    Covers shunt_classification_and_ligation_llm.py LLM calls:
      - Shunt type classification (_call_llm_for_shunt_classification)
      - CHIVA ligation planning (_call_llm_for_ligation)
    """
    return Agent(
        role="CHIVA Shunt Classification and Ligation Specialist",
        goal=(
            "Classify the venous shunt type from CHIVA virtual clips, "
            "then produce a step-by-step CHIVA ligation plan that preserves the saphenous vein."
        ),
        backstory=(
            "Senior vascular surgeon specialising in CHIVA haemodynamic surgery. "
            "Maps EP/RP clip patterns to shunt types (1, 2A-2C, 3, 4, 5, 1+2) "
            "and recommends the minimal-invasive ligation sequence that avoids unnecessary ablation."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_general_medical_assistant() -> Agent:
    """
    Covers the direct groq_client call in routes/general.py:
      - General medical Q&A with RAG-retrieved context
    """
    return Agent(
        role="Medical Knowledge Assistant",
        goal=(
            "Answer general venous disease and CHIVA methodology questions "
            "using the provided knowledge base context. Plain text only, no markdown."
        ),
        backstory=(
            "Clinical medical assistant specialising in venous disease and CHIVA methodology. "
            "Answers concisely in plain ASCII text, cites sources, "
            "and refuses off-topic requests outside venous surgery."
        ),
        llm=_make_llm(),
        verbose=False,
        allow_delegation=False,
    )
