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
import random
import re
import threading
import time
from collections import deque

from groq import Groq, RateLimitError

import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WAIT_SECONDS_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)
_client = None
GROQ_RATE_LIMIT_MAX_RETRIES = 7  # this project's own backoff for 429s specifically --
# raised 3 -> 5 -> 7: confirmed on a real job that 5 retries could still ALL collide into
# a fresh 429 when many concurrent workers (Pass 2's Stage 3a/3b parallelism) share the
# same TPM ceiling and retry in near-lockstep -- fixed the lockstep itself with jitter
# (see wait_s below) AND gave extra headroom on top, since jitter reduces but doesn't
# guarantee zero collisions under heavy concurrency. Each retry still waits out Groq's
# own suggested time (see _WAIT_SECONDS_RE below) plus jitter, so this only costs real
# wall-clock time when actually rate-limited, not on every call.
# separate from the SDK's max_retries=1 below, which covers generic transient errors.
# A 429 is fully expected under real usage (confirmed live: a production job hit
# "Rate limit reached ... tokens per minute (TPM)" and crashed at 75% progress because
# nothing above call_vlm_json ever retried it -- jobs.py's exception handler just marks
# the whole job "error" on any exception, no partial-progress recovery). Waiting out the
# limit and retrying is the correct response to a 429 (it is not a real failure, just a
# "not yet" from Groq), so this belongs here rather than in every caller.


class _TokenRateLimiter:
    """Thread-safe sliding-60s-window limiter for one of Groq's real per-minute token
    ceilings for this model/account. Two SEPARATE ceilings are confirmed real from actual
    429s, and BOTH need their own instance of this class, tracked independently (see
    _otpm_limiter / _tpm_limiter below) -- an earlier version of this fix only had the
    OTPM one, which left TPM completely unguarded and still crashed a real job once Pass 2
    started sending images concurrently:
      - OTPM (output-tokens-per-minute): 32000. "Rate limit reached ... on output tokens
        per minute (OTPM): Limit 32000, Used 31472, Requested 15754."
      - TPM (total tokens-per-minute, input+output): 250000. "Rate limit reached ... on
        tokens per minute (TPM): Limit 250000, Used 247907, Requested 7784."

    Concurrency (config.STAGE*_MAX_WORKERS) is a GUESS at a safe worker count; it has no
    way to know how close the real usage is to either ceiling until it's already over,
    which is exactly why raising it kept reproducing 429 storms instead of just going
    faster. This replaces guessing with actually tracking spend: every call RESERVES an
    estimated token cost against EACH relevant ceiling before it's allowed to fire
    (blocking/sleeping if the rolling window is too full), then CORRECTS that reservation
    to the call's real usage once the response comes back -- so a call that used far less
    than its estimate frees that headroom immediately for the next caller instead of
    holding it hostage for the full 60s. This is what lets concurrency actually help
    (workers wait here, safely, instead of firing and eating a 429) rather than just being
    a different way to guess wrong.
    """
    def __init__(self, limit_per_min: int):
        self._limit = limit_per_min
        self._lock = threading.Lock()
        self._events = deque()  # each entry: [timestamp, token_cost] (mutable for correct())

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > 60.0:
            self._events.popleft()

    def reserve(self, estimate: int) -> list:
        """Blocks until `estimate` tokens fit under the rolling budget, reserves them,
        returns the mutable entry (pass to correct() once actual usage is known)."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(cost for _, cost in self._events)
                if used + estimate <= self._limit:
                    entry = [now, estimate]
                    self._events.append(entry)
                    return entry
                # Not enough headroom right now. Poll at a short, bounded interval rather
                # than sleeping the full time until the oldest reservation ages out --
                # correct() calls from OTHER in-flight callers can free up real headroom
                # (an over-estimated reservation dropping to its much lower actual usage)
                # at any moment, and a single long sleep would miss that entirely, waiting
                # out the full 60s window even when the budget freed up in the first
                # couple seconds (confirmed: this exact bug showed an 8th concurrent
                # caller waiting 60s+ despite 6 of the other 7 correcting down to <2000
                # actual tokens almost immediately after their reservation).
                wait_s = min(max(60.0 - (now - self._events[0][0]) + 0.5, 0.5), 2.0) if self._events else 1.0
            time.sleep(wait_s)

    def correct(self, entry: list, actual_tokens: int) -> None:
        with self._lock:
            entry[1] = actual_tokens


class _UsageTracker:
    """Thread-safe running total of real token usage + estimated cost for the current job
    -- confirmed real user need: a job that crashed after 40 minutes cost $4.50 with zero
    visibility into where it went. reset() at the start of each run_full_pipeline call so
    each job's total is reported independently rather than accumulating across jobs in
    the same server process."""
    def __init__(self):
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._calls = 0

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._calls += 1

    def summary(self) -> dict:
        with self._lock:
            p, c, n = self._prompt_tokens, self._completion_tokens, self._calls
        # Pricing per this project's own memory notes (qwen/qwen3.6-27b on Groq):
        # ~$0.60/M input tokens, ~$3.00/M output tokens -- approximate, Groq's published
        # rate had some source disagreement when this was last checked, so treat the
        # dollar figure as an estimate for tracking trend/scale, not a billing-accurate
        # number.
        est_cost = (p / 1_000_000) * 0.60 + (c / 1_000_000) * 3.00
        return {"calls": n, "prompt_tokens": p, "completion_tokens": c,
                "total_tokens": p + c, "est_cost_usd": round(est_cost, 3)}


usage_tracker = _UsageTracker()


# Reserve slightly below each real ceiling -- leaves a small margin for estimation error
# and for any other caller on this same API key outside this process.
_otpm_limiter = _TokenRateLimiter(limit_per_min=28000)    # real ceiling: 32000 OTPM
_tpm_limiter = _TokenRateLimiter(limit_per_min=220000)    # real ceiling: 250000 TPM


def _estimate_prompt_tokens(system_prompt: str, user_text: str, image_b64: str,
                             extra_images: list) -> int:
    """Rough pre-call estimate of INPUT tokens (text + image), used only to size the TPM
    reservation before we know the real number -- corrected to the API's own reported
    prompt_tokens the moment the response comes back (see call_vlm_json), same
    self-healing pattern as the OTPM reservation. Deliberately generous per-image
    (vision-model image tokenization isn't documented precisely for this model via Groq,
    and this project's own real 429 evidence shows naming calls alone -- small text,
    ONE image -- requesting ~7800 tokens, meaning the image dominates the prompt-token
    cost far more than the text does) -- under-estimating here is what causes a real TPM
    violation to slip through uncorrected until the round-trip completes, so err high."""
    text_tokens = (len(system_prompt) + len(user_text)) // 4  # ~4 chars/token, rough
    # Calibrated against the real crash this fix targets: Groq's own error reported
    # "Requested 7784" for a stage3b_naming call (short text prompt, ONE image,
    # max_tokens=3072 default) -- working backward, ~7784 - 3072 - ~200(text) leaves
    # ~4500 tokens of real per-image cost, well above a first-guess estimate of 2000.
    # Erring high here is cheap (a brief extra queueing wait); erring low is what let the
    # real violation happen, so this is deliberately generous rather than a tight guess.
    image_tokens = 4500 if image_b64 else 0
    image_tokens += 3000 * len(extra_images or [])
    return text_tokens + image_tokens


def _get_client() -> Groq:
    global _client
    if _client is None:
        # max_retries=1 (SDK default is 2): a stuck/rate-limited call at
        # GROQ_TIMEOUT_SEC=60 with the default of 2 retries can silently burn up to
        # ~180s on ONE call before finally raising or succeeding, invisible to any
        # logging in this project (the retry/backoff happens inside the SDK's HTTP
        # layer). That compounding tail latency across dozens of sequential Pass-2
        # calls is the leading suspect for "job took far longer than expected" reports
        # — capping at 1 retry still absorbs a genuine one-off transient error/429
        # but roughly halves the worst-case stall per call. Not zero, since a call
        # failing outright (no retry) would just move the same problem to
        # pipeline.py's per-tick error handling instead of fixing it.
        _client = Groq(api_key=config.GROQ_API_KEY, max_retries=1)
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
    label: str = None,
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

    # Visible timing per call, tagged with a caller-supplied label — this is the
    # single biggest diagnostic gap that made "why is this slow" unanswerable before:
    # the SDK's own retry/backoff on rate limits happens silently inside
    # client.chat.completions.create(), so without a wall-clock stamp around the call
    # a 3s call and a 90s-after-two-retries call look identical in the pipeline's own
    # logs. Printed (not logged via `logging`) to match this project's existing
    # print()-based console output convention (see pipeline.py).
    tag = label or "groq_call"
    # Reserve the call's own max_tokens ceiling against the real OTPM budget BEFORE
    # firing -- see _OtpmLimiter. This MUST be the actual max_tokens passed to Groq, not a
    # smaller guess: Groq enforces on real generated tokens, so a call can never produce
    # more than its own max_tokens, which means reserving exactly that value is the only
    # way to GUARANTEE this process's own calls can never collectively exceed the budget
    # (a smaller estimate was tried first and confirmed insufficient -- it still let rare
    # 429s through, because another call could start believing there was room during the
    # window before a big call's actual usage got corrected in). Callers control real
    # concurrency by passing a smaller max_tokens for calls that don't need the full
    # ceiling (see stage2_fascia_classify._first_attempt_max_tokens) rather than this
    # wrapper under-reserving on their behalf. correct() below still shrinks the
    # reservation to actual usage the moment it's known, so an over-generous max_tokens
    # only costs a brief, self-healing wait for other callers, never a stuck reservation.
    effective_max = max_tokens or config.GROQ_MAX_TOKENS
    reservation = _otpm_limiter.reserve(effective_max)
    # TPM (total tokens: input+output) is a SEPARATE real ceiling from OTPM -- confirmed
    # by a real production crash where Pass 2's newly-concurrent naming calls (each
    # sending a full annotated image) burned through the 250000 TPM budget while staying
    # comfortably under the 32000 OTPM one, so the OTPM-only reservation above provided
    # zero protection against it. See _estimate_prompt_tokens for the pre-call estimate.
    prompt_estimate = _estimate_prompt_tokens(system_prompt, user_text, image_b64, extra_images)
    tpm_reservation = _tpm_limiter.reserve(prompt_estimate + effective_max)

    t0 = time.monotonic()
    resp = None
    try:
        for attempt in range(GROQ_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except RateLimitError as exc:
                if attempt >= GROQ_RATE_LIMIT_MAX_RETRIES:
                    elapsed = time.monotonic() - t0
                    print(f"[groq] {tag} FAILED after {elapsed:.1f}s ({attempt} rate-limit "
                          f"retries exhausted): {exc}")
                    raise
                m = _WAIT_SECONDS_RE.search(str(exc))
                # Groq's own error message names the exact wait -- use it plus a small
                # margin rather than a guessed backoff schedule. Falls back to a
                # conservative fixed wait if the message format ever changes. Random
                # jitter added on top -- confirmed real failure mode without it: many
                # concurrent workers all rate-limited by the same shared TPM ceiling
                # naturally converge on nearly the SAME short wait (Groq's suggested
                # delay is usually just 1-2s), so they retry in near-lockstep and
                # re-collide into another 429 together, burning through all
                # GROQ_RATE_LIMIT_MAX_RETRIES attempts as a group instead of spreading
                # out -- confirmed on a real job where every one of 5 retries still hit
                # a fresh 429 and the whole job crashed. Jitter desynchronizes that.
                wait_s = (float(m.group(1)) + 1.0 if m else 15.0) + random.uniform(0.0, 2.0)
                print(f"[groq] {tag} rate-limited (attempt {attempt + 1}/"
                      f"{GROQ_RATE_LIMIT_MAX_RETRIES}), waiting {wait_s:.1f}s: {exc}")
                # A 429 despite our own reservation means our estimate/limiter is out of
                # sync with reality (e.g. another process sharing this key, or this call's
                # true cost exceeded our pre-call estimate) -- correct BOTH reservations
                # up to a conservative value so we stop under-reserving for the retry,
                # then wait Groq's own suggested time (plus jitter) on top of that.
                _otpm_limiter.correct(reservation, effective_max)
                _tpm_limiter.correct(tpm_reservation, prompt_estimate + effective_max)
                time.sleep(wait_s)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                print(f"[groq] {tag} FAILED after {elapsed:.1f}s: {exc}")
                raise
    finally:
        if resp is not None and getattr(resp, "usage", None) is not None:
            _otpm_limiter.correct(reservation, resp.usage.completion_tokens)
            _tpm_limiter.correct(tpm_reservation,
                                  getattr(resp.usage, "total_tokens", None)
                                  or (resp.usage.prompt_tokens + resp.usage.completion_tokens))
            usage_tracker.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        elif resp is None:
            # Call never succeeded (raised) -- release both reservations rather than
            # holding phantom budget hostage for 60s on a call that produced no tokens.
            _otpm_limiter.correct(reservation, 0)
            _tpm_limiter.correct(tpm_reservation, 0)
    elapsed = time.monotonic() - t0
    raw = resp.choices[0].message.content or ""
    if elapsed > 8.0:
        # Anything this slow on a single call is either heavy reasoning (expected for
        # reasoning_effort="default") or a silent retry-after-rate-limit having
        # happened underneath — flagged so a long job's console output shows exactly
        # which calls/timestamps are the actual bottleneck instead of a single opaque
        # total at the end.
        print(f"[groq] {tag} took {elapsed:.1f}s (slow) — {len(raw)} chars returned")
    else:
        print(f"[groq] {tag} took {elapsed:.1f}s")
    return extract_json(raw), raw
