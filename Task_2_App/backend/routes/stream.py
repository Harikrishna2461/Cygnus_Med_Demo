"""
WebSocket event handlers for the continuous streaming guidance mode.

Events (client → server):
  stream_start  — create/reset session
  probe_move    — new probe position from leg diagram
  clip_mark     — surgeon marks an EP/RP finding
  clips_replace — surgeon deleted a clip (frontend sends full updated list)
  session_end   — surgeon ends the session

Events (server → client):
  session_ready   — session created, ready to stream
  guidance_update — new guidance + VLM + thinking entry
  clips_updated   — clip list changed (after mark or delete)
  session_ended   — session finished
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

NS = "/"  # default namespace — must match socket.io client default


def register_stream_events(socketio) -> None:
    """Register all SocketIO event handlers. Called once from app.py."""

    import streaming_session as sess_store
    import streaming_guidance_engine as engine

    # ── stream_start ──────────────────────────────────────────────────────────

    @socketio.on("stream_start")
    def handle_stream_start(data: dict):
        from flask_socketio import emit
        session_id = str(data.get("session_id", "default"))
        sess = sess_store.get_or_create(session_id)
        sess.clips.clear()
        sess.history.clear()
        sess.thinking_log.clear()
        sess.scan_log.clear()
        sess.generation        = 0
        sess.active            = True
        sess.last_vlm_pos_y    = -1.0
        sess.last_vlm_summary  = "No frame analyzed yet."
        sess.last_llm_pos_y    = -1.0
        sess.max_visited_pos_y = 0.0
        logger.info("Stream session started: %s", session_id)
        emit("session_ready", {"session_id": session_id})

    # ── probe_move ────────────────────────────────────────────────────────────

    @socketio.on("probe_move")
    def handle_probe_move(data: dict):
        from flask import request
        session_id = str(data.get("session_id", "default"))
        sess = sess_store.get_or_create(session_id)
        if not sess.active:
            return

        client_sid = request.sid
        # Snapshot probe position for this request
        region   = data.get("region", "UNKNOWN")
        pos_y    = float(data.get("pos_y_ratio", 0.0))
        pos_x    = float(data["pos_x_ratio"]) if data.get("pos_x_ratio") is not None else None
        surface  = data.get("surface", "anterior-medial")
        leg      = data.get("leg", "right")
        is_front = bool(data["is_front"]) if data.get("is_front") is not None else None

        # Record this request's generation BEFORE spawning thread
        req_gen = sess.bump()

        def process():
            try:
                result = engine.process_probe_state(
                    session=sess,
                    region=region,
                    pos_y=pos_y,
                    surface=surface,
                    leg=leg,
                    pos_x=pos_x,
                    is_front=is_front,
                )
            except Exception as exc:
                logger.error("process_probe_state error: %s", exc, exc_info=True)
                result = {
                    "guidance": "Server error — check backend logs.",
                    "vlm": None, "vlm_summary": "", "region": region,
                    "pos_y": pos_y, "thinking": None,
                }

            # Only skip if a significantly newer request has arrived AND it's
            # already been processed.  Use a looser check: discard only if
            # generation advanced by more than 6 (prevents last-result starvation).
            if sess.generation > req_gen + 6:
                return

            # Nothing new computed — don't spam the client
            if result.get("guidance") is None and result.get("vlm") is None:
                return

            socketio.emit("guidance_update", result, room=client_sid, namespace=NS)

        threading.Thread(target=process, daemon=True).start()

    # ── clip_mark ─────────────────────────────────────────────────────────────

    @socketio.on("clip_mark")
    def handle_clip_mark(data: dict):
        from flask import request
        from flask_socketio import emit

        session_id = str(data.get("session_id", "default"))
        sess = sess_store.get_or_create(session_id)
        # Allow marking even if sess.active is False (surgeon may mark after scan)
        # so we don't gate on active here.

        client_sid = request.sid

        clip = {
            "flow":             data.get("flow", "EP"),
            "from_type":        data.get("from_type", "N1"),
            "to_type":          data.get("to_type", "N2"),
            "pos_y_ratio":      float(data.get("pos_y_ratio", 0.0)),
            "leg":              data.get("leg", "right"),
            "region":           data.get("region", "UNKNOWN"),
            "elimination_test": data.get("elimination_test", ""),
        }
        sess.clips.append(clip)
        logger.info("Clip marked: %s  session=%s", clip, session_id)

        # Confirm clip list to client immediately
        emit("clips_updated", {"clips": sess.clips})

        # Force a new LLM call so guidance reflects the newly confirmed finding
        region  = data.get("region", "UNKNOWN")
        pos_y   = float(data.get("pos_y_ratio", 0.0))
        surface = data.get("surface", "anterior-medial")
        leg     = data.get("leg", "right")
        req_gen = sess.bump()

        def process_after_mark():
            try:
                result = engine.process_probe_state(
                    session=sess,
                    region=region, pos_y=pos_y,
                    surface=surface, leg=leg,
                    force_llm=True,
                )
            except Exception as exc:
                logger.error("LLM after clip_mark: %s", exc, exc_info=True)
                return
            socketio.emit("guidance_update", result, room=client_sid, namespace=NS)

        threading.Thread(target=process_after_mark, daemon=True).start()

    # ── clips_replace (surgeon deleted a clip) ────────────────────────────────

    @socketio.on("clips_replace")
    def handle_clips_replace(data: dict):
        from flask_socketio import emit
        session_id = str(data.get("session_id", "default"))
        sess = sess_store.get_or_create(session_id)
        new_clips = data.get("clips", [])
        sess.clips = new_clips
        emit("clips_updated", {"clips": sess.clips})

    # ── session_end ───────────────────────────────────────────────────────────

    @socketio.on("session_end")
    def handle_session_end(data: dict):
        from flask_socketio import emit
        session_id = str(data.get("session_id", "default"))
        sess = sess_store.get(session_id)
        if sess:
            sess.active = False
        logger.info("Stream session ended: %s", session_id)
        emit("session_ended", {
            "session_id": session_id,
            "clips":      sess.clips if sess else [],
            "log_count":  len(sess.thinking_log) if sess else 0,
        })
