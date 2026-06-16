"""
Per-session state for continuous streaming guidance.

Each session holds:
  - clips       : EP/RP marks confirmed by the surgeon so far
  - history     : rolling LLM conversation (list of Groq message dicts)
  - thinking_log: every (state_msg, raw_response) pair for the UI log
  - generation  : monotone counter — only the latest probe_move emits
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


_global_lock = threading.Lock()
_sessions: dict[str, "StreamSession"] = {}


@dataclass
class StreamSession:
    session_id: str
    clips: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)   # Groq message dicts
    thinking_log: list[dict] = field(default_factory=list)
    generation: int = 0
    active: bool = True

    # Rate-limit state — avoid calling VLM/LLM on every mouse pixel
    last_vlm_pos_y: float = -1.0
    last_vlm_summary: str = "No frame analyzed yet."
    last_llm_pos_y: float = -1.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── generation counter ────────────────────────────────────────────────────

    def bump(self) -> int:
        """Increment generation and return the new value."""
        with self._lock:
            self.generation += 1
            return self.generation

    def is_current(self, gen: int) -> bool:
        with self._lock:
            return self.generation == gen

    # ── history management ────────────────────────────────────────────────────

    def push_exchange(self, user_msg: str, assistant_msg: str, window: int = 8) -> None:
        """Append one (user, assistant) pair and trim to window size."""
        self.history.append({"role": "user",      "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        cap = window * 2
        if len(self.history) > cap:
            self.history = self.history[-cap:]

    def push_thinking(self, pos_y: float, region: str, state_msg: str, raw: str, guidance: str) -> None:
        self.thinking_log.append({
            "pos_y":    round(pos_y, 2),
            "region":   region,
            "state":    state_msg,
            "raw":      raw,
            "guidance": guidance,
        })
        # Keep only the last 50 entries to avoid unbounded growth
        if len(self.thinking_log) > 50:
            self.thinking_log = self.thinking_log[-50:]


# ── session registry ──────────────────────────────────────────────────────────

def get_or_create(session_id: str) -> StreamSession:
    with _global_lock:
        if session_id not in _sessions:
            _sessions[session_id] = StreamSession(session_id=session_id)
        return _sessions[session_id]


def get(session_id: str) -> Optional[StreamSession]:
    with _global_lock:
        return _sessions.get(session_id)


def remove(session_id: str) -> None:
    with _global_lock:
        _sessions.pop(session_id, None)
