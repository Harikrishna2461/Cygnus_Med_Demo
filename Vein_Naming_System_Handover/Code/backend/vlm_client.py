"""
Local VLM/LLM client -- the single choke point through which EVERY reasoning/vision call in
this project goes (Stage 2 N1/N2/N3 classification, Stage 3a probe location, Stage 3b vein
naming, and the ROI-crop VLM helper).

This replaces the original hosted-API wrapper. The model is now Qwen (27B, 4-bit GGUF,
Q4_K_M + mmproj-F16 vision projector) served locally by llama.cpp's `llama-server`, which
exposes an OpenAI-compatible HTTP API. See the handover Developer Guide for how it is
launched (start_llama_server.bat / docker-compose.yml).

The public surface deliberately matches what the pipeline modules already call
(`call_vlm_json(...)`, `usage_tracker`, `extract_json`, `strip_think`) so the stage modules
only changed their import line. What is different underneath:

  * No token-per-minute rate limiter, no 429 retry storm handling -- a local server has no
    per-minute quota. It has a different real constraint instead: a fixed number of decode
    "slots" (llama-server --parallel N) that each own a slice of the context window. A
    semaphore sized to that slot count (config.VLM_MAX_CONCURRENT) keeps callers queueing
    politely here rather than piling requests onto the server.
  * `reasoning_effort` ("none" | "default") is mapped onto the Qwen chat-template switch
    `enable_thinking` (False | True). config.VLM_FORCE_NO_THINKING (default True) overrides
    every call to thinking OFF -- the local evaluation showed it is faster AND more accurate.
  * llama-server returns the reasoning trace in `message.reasoning_content` and only the
    final answer in `message.content`. JSON is parsed from `content` only, never from the
    reasoning trace, so stray braces in mid-reasoning prose can't be mistaken for output.
  * Cost tracking is gone (local inference is free); the tracker still counts calls/tokens
    so a job's summary line stays useful for judging load.
"""
import json
import re
import threading
import time

import requests

import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_slots = threading.BoundedSemaphore(config.VLM_MAX_CONCURRENT)
_session = requests.Session()


class _UsageTracker:
    """Thread-safe running total of real token usage for the current job. reset() at the
    start of each run_full_pipeline call so each job reports independently."""
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._calls = 0
            self._busy_seconds = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        with self._lock:
            self._prompt_tokens += prompt_tokens or 0
            self._completion_tokens += completion_tokens or 0
            self._calls += 1
            self._busy_seconds += seconds

    def summary(self) -> dict:
        with self._lock:
            p, c, n, s = self._prompt_tokens, self._completion_tokens, self._calls, self._busy_seconds
        return {"calls": n, "prompt_tokens": p, "completion_tokens": c,
                "total_tokens": p + c, "cumulative_call_seconds": round(s, 1)}


usage_tracker = _UsageTracker()


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


def server_status() -> dict:
    """Polls llama-server's /health. {"state": "ready" | "loading" | "not_started" | "error"}."""
    try:
        data = _session.get(f"{config.LLAMA_SERVER_URL}/health", timeout=5).json()
        state = {"ok": "ready", "loading model": "loading"}.get(data.get("status"), "error")
        return {"state": state, "detail": data}
    except requests.exceptions.ConnectionError:
        return {"state": "not_started", "detail": None}
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "detail": str(exc)}


def wait_until_ready(timeout_sec: float = None) -> None:
    """Blocks until llama-server reports ready (model + mmproj resident on the GPU).
    Raises RuntimeError with an actionable message on timeout -- called once at the start of
    a job so a forgotten `start_llama_server` fails in seconds, not after ROI cropping and
    a full segmentation pass."""
    deadline = time.monotonic() + (timeout_sec if timeout_sec is not None else config.VLM_STARTUP_WAIT_SEC)
    last = None
    while True:
        last = server_status()
        if last["state"] == "ready":
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Local Qwen server at {config.LLAMA_SERVER_URL} is not ready (state="
                f"{last['state']}). Start it first: see start_llama_server.bat / "
                f"docker-compose.yml. Detail: {last['detail']}")
        time.sleep(2.0)


def _build_content(user_text: str, image_b64: str, image_media_type: str, extra_images: list) -> list:
    content = []
    if image_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}})
    for ref_b64, ref_media_type in (extra_images or []):
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{ref_media_type};base64,{ref_b64}"}})
    content.append({"type": "text", "text": user_text})
    return content


def _chat(messages: list, max_tokens: int, temperature: float, timeout: float,
          enable_thinking: bool, label: str) -> tuple[str, str]:
    """One chat-completions round trip with bounded retry on transient errors.
    Returns (final_answer_text, reasoning_text)."""
    payload = {
        "messages": messages,
        "max_tokens": min(max_tokens, config.VLM_MAX_TOKENS_CAP),
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    attempts = config.VLM_TRANSIENT_RETRIES + 1
    with _slots:  # wait for a free decode slot; time spent queued is not counted as call time
        t0 = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                resp = _session.post(f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
                                     json=payload, timeout=timeout)
                if resp.status_code >= 500 or resp.status_code == 503:
                    # 503 = "Loading model" / all slots busy. Both are worth a short retry.
                    raise requests.exceptions.ConnectionError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt >= attempts:
                    print(f"[vlm] {label} FAILED after {time.monotonic() - t0:.1f}s "
                          f"({attempt} attempt(s)): {exc}")
                    raise
                wait_s = 3.0 * attempt
                print(f"[vlm] {label} transient error (attempt {attempt}/{attempts}), "
                      f"retrying in {wait_s:.0f}s: {exc}")
                time.sleep(wait_s)
        elapsed = time.monotonic() - t0

    choice = data["choices"][0]
    message = choice["message"]
    answer = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    usage = data.get("usage", {})
    usage_tracker.add(usage.get("prompt_tokens"), usage.get("completion_tokens"), elapsed)

    if choice.get("finish_reason") == "length":
        # Real, documented failure mode of a thinking model: the whole budget got spent
        # reasoning and the answer never started. Loud on purpose -- callers already
        # treat an empty answer as "this tick failed" and retry with a bigger budget.
        print(f"[vlm] {label} HIT max_tokens={payload['max_tokens']} "
              f"(answer chars={len(answer)}, reasoning chars={len(reasoning)}) -- output truncated")
    tag = "(slow) " if elapsed > 8.0 else ""
    print(f"[vlm] {label} took {elapsed:.1f}s {tag}-- {usage.get('completion_tokens')} completion tokens")
    return answer, reasoning


def call_vlm_json(
    system_prompt: str,
    user_text: str,
    image_b64: str = None,
    image_media_type: str = "image/png",
    extra_images: list[tuple[str, str]] = None,
    model: str = None,        # accepted for signature compatibility; the server hosts one model
    max_tokens: int = None,
    temperature: float = None,
    timeout: float = None,
    reasoning_effort: str = None,
    label: str = None,
) -> tuple[dict, str]:
    """Returns (parsed_json, raw_answer_text).

    parsed_json is {} if the reply had no valid JSON object -- callers must treat that as
    "this tick failed", never substitute a guess. Connection/HTTP errors that survive the
    bounded retry are raised; that is a pipeline-level skip decision, not this wrapper's.

    extra_images: optional (b64, media_type) tuples sent after the primary image, in the
    same OpenAI-style multi-part content list (llama-server accepts multiple image_url
    parts when started with --mmproj)."""
    effort = config.VLM_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    if config.VLM_FORCE_NO_THINKING:
        effort = "none"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_content(user_text, image_b64, image_media_type, extra_images)},
    ]
    answer, _reasoning = _chat(
        messages,
        max_tokens=max_tokens or config.VLM_MAX_TOKENS,
        temperature=config.VLM_TEMPERATURE if temperature is None else temperature,
        timeout=timeout or config.VLM_TIMEOUT_SEC,
        enable_thinking=(effort == "default"),
        label=label or "vlm_call",
    )
    return extract_json(answer), answer


def call_vlm_text(
    system_prompt: str,
    user_text: str,
    image_b64: str = None,
    image_media_type: str = "image/jpeg",
    max_tokens: int = 150,
    reasoning_effort: str = "none",
    label: str = None,
) -> str:
    """Plain-text variant (used by vlm_agent.py's ROI / view-type helper, which does its own
    parsing). Returns the final answer text, stripped."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_content(user_text, image_b64, image_media_type, None)},
    ]
    answer, _ = _chat(messages, max_tokens=max_tokens, temperature=config.VLM_TEMPERATURE,
                      timeout=config.VLM_TIMEOUT_SEC,
                      enable_thinking=(reasoning_effort == "default" and not config.VLM_FORCE_NO_THINKING),
                      label=label or "vlm_text")
    return strip_think(answer)
