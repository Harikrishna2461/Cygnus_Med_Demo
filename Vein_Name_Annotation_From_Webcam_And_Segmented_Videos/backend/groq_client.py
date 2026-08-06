"""
Thin Groq wrapper for the VLM calls (qwen/qwen3.6-27b).

reasoning_effort="none" (config.GROQ_REASONING_EFFORT, same setting already used by the
copied ROI_Identification agents) turns off the model's separate verbose <think> preamble
— it still makes the classification/naming call itself, just without spending a large
chunk of latency generating visible chain-of-thought tokens first. This was the dominant
cost in end-to-end runtime (a 20s test clip took ~3 minutes, almost entirely thinking
tokens). strip_think() is kept as a harmless no-op safety net in case a caller ever passes
a different reasoning_effort that does emit a <think> block.
"""
import json
import re

from groq import Groq

import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_json(text: str) -> dict:
    text = strip_think(text)
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def call_vlm_json(
    system_prompt: str,
    user_text: str,
    image_b64: str = None,
    image_media_type: str = "image/png",
    extra_images: list[tuple[str, str]] = None,
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    timeout: float = None,
    reasoning_effort: str = None,
) -> tuple[dict, str]:
    """Returns (parsed_json, raw_response_text).

    parsed_json is {} if the model's reply had no valid JSON object — callers must treat
    that as "this tick's classification failed", not silently substitute a guess.
    Network/API errors are not caught here; that's a pipeline-level retry/skip decision,
    not this wrapper's job.

    extra_images: optional list of (b64, media_type) tuples for additional reference
    images (e.g. a labeled example frame) sent alongside the primary image_b64 — appended
    to the same OpenAI-compatible multi-part `content` list, which Groq's vision models
    accept multiple image_url entries in. Put after the primary image so the caller's
    text can refer to "the frame above" vs. "the reference example below" unambiguously.
    """
    client = _get_client()
    content = []
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"},
        })
    for ref_b64, ref_media_type in (extra_images or []):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ref_media_type};base64,{ref_b64}"},
        })
    content.append({"type": "text", "text": user_text})

    kwargs = dict(
        model=model or config.GROQ_VLM_MODEL,
        max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
        temperature=config.GROQ_TEMPERATURE if temperature is None else temperature,
        timeout=timeout or config.GROQ_TIMEOUT_SEC,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )
    effort = config.GROQ_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    # "default" is a sentinel meaning "don't send reasoning_effort at all" — lets the
    # model use its own full chain-of-thought rather than the fast/no-think mode. Needed
    # for calls doing genuinely multi-step reasoning (see stage3_webcam_location.py's
    # facing-direction mirroring) where "none" measurably hurt consistency — confirmed on
    # real footage: the same near-identical frame ("patient facing camera, probe on
    # right side of image") got mirrored correctly sometimes and backwards other times
    # across adjacent 2s ticks with reasoning off.
    if effort != "default":
        kwargs["reasoning_effort"] = effort

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    return extract_json(raw), raw
