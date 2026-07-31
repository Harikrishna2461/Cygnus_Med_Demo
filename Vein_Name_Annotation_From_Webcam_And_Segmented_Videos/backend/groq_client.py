"""
Thin Groq wrapper for the reasoning VLM calls (qwen/qwen3.6-27b).

Unlike the /no_think one-word-answer call already used elsewhere in this codebase
(Task_4_VLM_Fascia_Vein_Detection/app.py's blob evaluator), our calls want the model to
actually reason, so responses carry a <think>...</think> block that must be stripped
before JSON parsing.
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
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    timeout: float = None,
) -> tuple[dict, str]:
    """Returns (parsed_json, raw_response_text).

    parsed_json is {} if the model's reply had no valid JSON object — callers must treat
    that as "this tick's classification failed", not silently substitute a guess.
    Network/API errors are not caught here; that's a pipeline-level retry/skip decision,
    not this wrapper's job.
    """
    client = _get_client()
    content = []
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"},
        })
    content.append({"type": "text", "text": user_text})

    resp = client.chat.completions.create(
        model=model or config.GROQ_VLM_MODEL,
        max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
        temperature=config.GROQ_TEMPERATURE if temperature is None else temperature,
        timeout=timeout or config.GROQ_TIMEOUT_SEC,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )
    raw = resp.choices[0].message.content or ""
    return extract_json(raw), raw
