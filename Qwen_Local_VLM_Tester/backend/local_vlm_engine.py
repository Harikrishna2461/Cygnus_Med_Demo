"""Talks to a locally-running `llama-server` (llama.cpp) process serving the GGUF build
of the model over its OpenAI-compatible HTTP API, and times every call.

Switched from an in-process transformers load (see git history / session notes) after
the compressed-tensors AWQ build turned out to fully decompress to ~55GB in VRAM on
first inference (no efficient int4 GEMM kernel for this hardware) and OOM'd the 32GB
5090. GGUF's int4 kernels are mature and genuinely keep weights compressed in VRAM.

llama-server itself is started separately (see run_llama_server.py / README) -- this
module is just a thin HTTP client + timer, same shape as groq_client.py in the real
pipeline so the two are easy to compare.
"""
import json
import re
import time

import requests

import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_json(text: str) -> dict | None:
    text = strip_think(text)
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def status() -> dict:
    """Polls llama-server's own /health endpoint -- it reports "loading model" until
    the GGUF (and mmproj) are fully resident on the GPU, then "ok"."""
    try:
        resp = requests.get(f"{config.LLAMA_SERVER_URL}/health", timeout=5)
        data = resp.json()
        state = {"ok": "ready", "loading model": "loading"}.get(data.get("status"), "error")
        return {"state": state, "error": None if state != "error" else data}
    except requests.exceptions.ConnectionError:
        return {"state": "not_started", "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "error": str(exc)}


def run_inference(
    system_prompt: str,
    user_text: str,
    image_b64_list: list[str],
    max_new_tokens: int = None,
    temperature: float = None,
    enable_thinking: bool = True,
) -> dict:
    """Returns dict with raw_text, parsed_json, elapsed_seconds, prompt_tokens,
    completion_tokens, tokens_per_second, error -- same shape the old transformers-based
    engine returned, so app.py needed no changes."""
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        for b64 in image_b64_list
    ]
    content.append({"type": "text", "text": user_text})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    payload = {
        "messages": messages,
        "max_tokens": max_new_tokens or config.MAX_NEW_TOKENS_DEFAULT,
        "temperature": temperature if temperature is not None else config.TEMPERATURE_DEFAULT,
        # This model reasons at length before answering by default (matches the real
        # pipeline's reasoning_effort="default" for the harder calls -- see
        # stage3_webcam_location.py's comments on why "none" hurt consistency there).
        # Confirmed real failure mode here: with thinking on and a low max_tokens, the
        # model can burn the entire budget mid-reasoning and never reach the final
        # answer, leaving content empty. Exposed as a toggle rather than hardcoded off,
        # since comparing thinking vs. non-thinking latency/quality is the point of this
        # tester.
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }

    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=config.REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 -- surfaced to caller, this is a test harness
        return {"error": str(exc)}
    elapsed = time.monotonic() - t0

    message = data["choices"][0]["message"]
    # This model "thinks" by default -- llama-server splits that out into
    # reasoning_content, separate from the final content -- fold both into raw_text
    # (clearly labeled) so a low max_new_tokens that truncates mid-reasoning still shows
    # something instead of silently returning empty text.
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    raw_text = (f"[reasoning]\n{reasoning}\n\n[answer]\n{content}" if reasoning else content)
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    return {
        "raw_text": raw_text,
        "parsed_json": extract_json(content),  # JSON only ever lives in the final answer,
        # never the reasoning trace -- parsing raw_text risks picking up stray braces
        # from mid-reasoning prose instead.
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(completion_tokens / elapsed, 2) if completion_tokens and elapsed > 0 else None,
        "error": None,
    }
