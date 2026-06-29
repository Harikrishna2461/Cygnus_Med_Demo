"""
Per-session state for continuous streaming guidance.

Each session holds:
  - clips        : EP/RP marks confirmed by the surgeon so far
  - history      : rolling LLM conversation (list of Groq message dicts)
  - thinking_log : every (state_msg, raw_response) pair for the UI log
  - scan_log     : lightweight record of every probe position visited this session
                   used by history_agent to build the posY-band coverage summary
  - generation   : monotone counter — only the latest probe_move emits
"""
from __future__ import annotations

import time
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
    # Lightweight positional record for history_agent coverage summary.
    # Each entry: {pos_y, region, surface, leg, t (unix timestamp)}
    scan_log: list[dict] = field(default_factory=list)
    generation: int = 0
    active: bool = True

    # Rate-limit state — avoid calling VLM/LLM on every mouse pixel
    last_vlm_pos_y: float = -1.0
    last_vlm_summary: str = "No frame analyzed yet."
    last_llm_pos_y: float = -1.0

    # Shunt types already confirmed and notified this session — prevents
    # re-firing the shunt_confirmed event for the same type on each probe_move.
    confirmed_shunts: list = field(default_factory=list)

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

    # ── scan log ──────────────────────────────────────────────────────────────

    def log_scan_position(
        self,
        pos_y: float,
        region: str,
        surface: str,
        leg: str,
        is_front: Optional[bool] = None,
    ) -> None:
        """Record a probe position visit for the history agent."""
        entry: dict = {
            "pos_y":   round(pos_y, 3),
            "region":  region,
            "surface": surface,
            "leg":     leg,
            "t":       round(time.time(), 2),
        }
        if is_front is not None:
            entry["is_front"] = is_front
        self.scan_log.append(entry)
        if len(self.scan_log) > 300:
            self.scan_log = self.scan_log[-300:]

    # ── history management ────────────────────────────────────────────────────

    def push_exchange(self, user_msg: str, assistant_msg: str, window: int = 8) -> None:
        """Append one (user, assistant) pair. window param reserved for future use."""
        self.history.append({"role": "user",      "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})

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
