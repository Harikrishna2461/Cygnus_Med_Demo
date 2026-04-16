"""
Sonographer profiles and session history.
Uses the same mlops_metrics.db as the main tracker.
"""
import sqlite3
import json
import uuid
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'mlops_metrics.db'

SEED_SONOGRAPHERS = [
    {
        "id": "sono-001",
        "name": "Dr. Sarah Chen",
        "title": "Senior Vascular Sonographer",
        "specialty": "Complex CHIVA & bilateral lower limb assessment",
        "experience_years": 12,
        "avatar_color": "#3b82f6",
        "scanning_style": (
            "Starts assessment at the SFJ (groin) and traces the GSV distally to the knee. "
            "Uses Valsalva manoeuvre routinely for valve competence testing. "
            "Prefers longitudinal views at SFJ before switching to transverse for diameter measurement. "
            "Tends to revisit reflux points twice to confirm duration. "
            "Typically scans the right leg first, then mirrors the pattern on the left."
        ),
    },
    {
        "id": "sono-002",
        "name": "Dr. James Okoye",
        "title": "Vascular Sonographer",
        "specialty": "Type 2 perforator mapping & calf vein assessment",
        "experience_years": 4,
        "avatar_color": "#059669",
        "scanning_style": (
            "Bottom-up scanning approach — starts at ankle perforators (Cockett zone) and works proximally. "
            "Thorough calf perforator mapping before assessing the GSV trunk. "
            "Spends more time confirming N3 tributary patterns; occasionally misses Hunterian perforators. "
            "Uses compression manoeuvre frequently in the Knee-Ankle zone. "
            "Logs additional notes on tributary branching and calibre."
        ),
    },
    {
        "id": "sono-003",
        "name": "Dr. Maria Santos",
        "title": "Lead Sonographer",
        "specialty": "Pelvic origin reflux, SSV & SPJ assessment",
        "experience_years": 8,
        "avatar_color": "#7c3aed",
        "scanning_style": (
            "Bilateral simultaneous comparison approach — assesses both legs in parallel before drawing conclusions. "
            "Starts posteriorly at the SPJ/popliteal fossa before moving to medial SFJ. "
            "Expert at detecting pelvic origin (P) reflux and complex Type 4/5 patterns. "
            "Routinely applies the elimination test whenever RP N3→N1 is seen alongside RP N2→N1. "
            "Documents SSV involvement carefully in SPJ-Ankle zone clips."
        ),
    },
]


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create sonographer tables and seed initial profiles if empty."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS sonographers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                title TEXT,
                specialty TEXT,
                experience_years INTEGER,
                avatar_color TEXT,
                scanning_style TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sonographer_sessions (
                session_id TEXT PRIMARY KEY,
                sonographer_id TEXT NOT NULL,
                session_date TIMESTAMP NOT NULL,
                mode TEXT,
                total_points INTEGER,
                reflux_count INTEGER,
                guidance_history TEXT,
                session_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(sonographer_id) REFERENCES sonographers(id)
            );
        """)
        cur.execute("SELECT COUNT(*) FROM sonographers")
        if cur.fetchone()[0] == 0:
            for s in SEED_SONOGRAPHERS:
                cur.execute(
                    """INSERT OR IGNORE INTO sonographers
                       (id, name, title, specialty, experience_years, avatar_color, scanning_style)
                       VALUES (?,?,?,?,?,?,?)""",
                    (s["id"], s["name"], s["title"], s["specialty"],
                     s["experience_years"], s["avatar_color"], s["scanning_style"]),
                )
            logger.info("Seeded 3 sonographer profiles")
    logger.info("Sonographer DB tables ready")


def get_all_sonographers():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT s.*,
                      COUNT(ss.session_id) AS session_count,
                      MAX(ss.session_date) AS last_session_date
               FROM sonographers s
               LEFT JOIN sonographer_sessions ss ON s.id = ss.sonographer_id
               GROUP BY s.id
               ORDER BY s.name"""
        )
        return [dict(r) for r in cur.fetchall()]


def get_sonographer(sono_id):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM sonographers WHERE id = ?", (sono_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_sessions(sono_id, limit=5):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM sonographer_sessions
               WHERE sonographer_id = ?
               ORDER BY session_date DESC LIMIT ?""",
            (sono_id, limit),
        )
        sessions = []
        for r in cur.fetchall():
            s = dict(r)
            s['guidance_history'] = json.loads(s.get('guidance_history') or '[]')
            sessions.append(s)
        return sessions


def save_session(sono_id, mode, guidance_history, session_summary):
    session_id = str(uuid.uuid4())
    reflux_count = sum(1 for h in guidance_history if h.get('flow_type') == 'RP')
    with get_db() as conn:
        conn.cursor().execute(
            """INSERT INTO sonographer_sessions
               (session_id, sonographer_id, session_date, mode, total_points,
                reflux_count, guidance_history, session_summary)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, sono_id, datetime.now().isoformat(), mode,
             len(guidance_history), reflux_count,
             json.dumps(guidance_history), session_summary),
        )
    return session_id


def build_sonographer_context(sono_id):
    """Return a text block for injection into LLM prompts."""
    sono = get_sonographer(sono_id)
    if not sono:
        return ""
    sessions = get_sessions(sono_id, limit=3)

    lines = [
        "=== SONOGRAPHER PROFILE (personalise guidance to this individual) ===",
        f"Name: {sono['name']} ({sono['title']})",
        f"Experience: {sono['experience_years']} years | Specialty: {sono['specialty']}",
        f"Habitual scanning style: {sono['scanning_style']}",
    ]

    # Add positional zone reference (normalized 0-1 coordinates)
    lines.extend([
        "\n=== ANATOMICAL SCANNING ZONES (normalized coordinates, top-left=(0,0), bottom-right=(1,1)) ===",
        "RIGHT LEG (left side of image):",
        "  • SFJ-Knee Zone: X=0.0931-0.475, Y=0-0.5497",
        "  • Knee-Ankle Zone: X=0.105-0.2947, Y=0.5497-1",
        "  • SPJ-Ankle Zone: X=0.2827-0.4386, Y=0.5497-1",
        "",
        "LEFT LEG (right side of image):",
        "  • SFJ-Knee Zone: X=0.4985-0.909, Y=0-0.5497",
        "  • Knee-Ankle Zone: X=0.7081-0.91, Y=0.5497-1",
        "  • SPJ-Ankle Zone: X=0.588-0.714, Y=0.5497-1",
    ])

    if sessions:
        lines.append(f"\n=== PREVIOUS SESSION HISTORY (last {len(sessions)}) ===")
        for i, s in enumerate(sessions, 1):
            date_str = s['session_date'][:10]
            lines.append(
                f"Session {i} ({date_str}): {s['total_points']} clips, "
                f"{s['reflux_count']} reflux detections, mode={s['mode']}"
            )
            if s.get('session_summary'):
                lines.append(f"  Summary: {s['session_summary']}")
            for h in s['guidance_history'][:2]:
                flow = h.get('flow_type', '?')
                inst = h.get('instruction', '')
                if inst:
                    lines.append(f"  [{flow}] {inst}")
    else:
        lines.append("\n(No previous sessions — first-time use for this sonographer)")

    lines.extend([
        "\nUSE THIS CONTEXT: Tailor your probe guidance to match this sonographer's experience level, "
        "known style, and past scanning patterns. Reference their habits where relevant.",
        "When providing probe position guidance, use the normalized coordinate zones above to be specific.",
    ])
    return "\n".join(lines)
