import sys
from pathlib import Path
from flask import Blueprint, redirect, send_file, session

bp = Blueprint("views", __name__)

if getattr(sys, "frozen", False):
    _exe_frontend = Path(sys.executable).parent / "frontend"
    _FRONTEND_DIR = _exe_frontend if _exe_frontend.exists() else Path(sys._MEIPASS) / "frontend"
else:
    _FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


def _require_login():
    """Returns a redirect if not logged in, else None."""
    if "user_id" not in session:
        return redirect("/login")
    return None


def _require_admin():
    """Returns a redirect/403 if not admin, else None."""
    if "user_id" not in session:
        return redirect("/login")
    if not session.get("is_admin"):
        return "Forbidden", 403
    return None


@bp.route("/login")
def login_page():
    if "user_id" in session:
        return redirect("/")
    return send_file(str(_FRONTEND_DIR / "login.html"))


@bp.route("/")
def landing():
    redir = _require_login()
    if redir:
        return redir
    is_admin = session.get("is_admin", False)
    username = session.get("username", "")
    admin_link = '<a href="/admin" style="color:#94a3b8;font-size:0.82em;text-decoration:none;">Admin Panel</a>' if is_admin else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Assistant</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a; min-height: 100vh;
            display: flex; align-items: center; justify-content: center; padding: 20px;
        }}
        .container {{ max-width: 900px; width: 100%; }}
        .topbar {{ display: flex; justify-content: flex-end; align-items: center; gap: 16px; margin-bottom: 24px; }}
        .topbar span {{ color: #94a3b8; font-size: 0.85em; }}
        .logout-btn {{
            background: transparent; border: 1px solid #334155; color: #94a3b8;
            padding: 6px 14px; border-radius: 6px; font-size: 0.82em; cursor: pointer;
            transition: all 0.2s;
        }}
        .logout-btn:hover {{ border-color: #475569; color: #cbd5e1; background: #1e293b; }}
        .header {{ text-align: center; margin-bottom: 60px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 12px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; }}
        .header p {{ font-size: 0.95em; color: #94a3b8; }}
        .modes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }}
        @media (max-width: 768px) {{
            .modes {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 2em; }}
        }}
        .mode-card {{
            background: #1e293b; border-radius: 12px; padding: 32px; cursor: pointer;
            transition: all 0.3s ease; border: 1px solid #334155;
            text-decoration: none; color: inherit; display: flex; flex-direction: column;
        }}
        .mode-card:hover {{ transform: translateY(-4px); border-color: #475569; background: #0f172a; }}
        .mode-icon {{ font-size: 2.5em; margin-bottom: 16px; }}
        .mode-card h2 {{ font-size: 1.5em; margin-bottom: 12px; color: #ffffff; font-weight: 600; }}
        .mode-card p {{ font-size: 0.9em; color: #cbd5e1; line-height: 1.6; margin-bottom: 20px; flex: 1; }}
        .features {{ list-style: none; margin-bottom: 20px; }}
        .features li {{ padding: 6px 0; color: #94a3b8; font-size: 0.85em; }}
        .features li:before {{ content: "- "; color: #64748b; margin-right: 8px; }}
        .btn {{ padding: 11px 20px; border-radius: 8px; font-size: 0.95em; font-weight: 600; text-decoration: none; transition: all 0.2s ease; border: none; cursor: pointer; align-self: flex-start; }}
        .btn-clinical, .btn-general {{ background: #2563eb; color: white; }}
        .btn-clinical:hover, .btn-general:hover {{ background: #1d4ed8; }}
        .footer {{ text-align: center; color: #64748b; font-size: 0.85em; margin-top: 50px; padding-top: 30px; border-top: 1px solid #334155; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="topbar">
            <span>Logged in as <strong style="color:#e2e8f0">{username}</strong></span>
            {admin_link}
            <button class="logout-btn" onclick="logout()">Log out</button>
        </div>
        <div class="header">
            <h1>Medical Assistant</h1>
            <p>Select a mode to begin</p>
        </div>
        <div class="modes">
            <a href="/clinical" class="mode-card">
                <div class="mode-icon">C</div>
                <h2>Clinical Support</h2>
                <p>Specialized analysis for venous shunt classification and ligation planning with clinical decision support.</p>
                <ul class="features">
                    <li>Shunt classification</li>
                    <li>Ligation guidance</li>
                    <li>Clinical reasoning</li>
                </ul>
                <button class="btn btn-clinical">Enter &rarr;</button>
            </a>
            <a href="/general" class="mode-card">
                <div class="mode-icon">+</div>
                <h2>General Chat</h2>
                <p>Ask any medical or surgical questions from the comprehensive knowledge base with intelligent search.</p>
                <ul class="features">
                    <li>Medical research</li>
                    <li>General knowledge</li>
                    <li>Evidence-based answers</li>
                </ul>
                <button class="btn btn-general">Enter &rarr;</button>
            </a>
        </div>
        <div class="footer">
            <p>Always consult clinical guidelines and specialists for critical decisions</p>
        </div>
    </div>
    <script>
    async function logout() {{
        await fetch('/api/logout', {{method:'POST'}});
        window.location.href = '/login';
    }}
    </script>
</body>
</html>"""


@bp.route("/clinical")
def clinical():
    redir = _require_login()
    if redir:
        return redir
    return send_file(str(_FRONTEND_DIR / "index.html"))


@bp.route("/general")
def general():
    redir = _require_login()
    if redir:
        return redir
    return send_file(str(_FRONTEND_DIR / "general.html"))


@bp.route("/admin")
def admin_panel():
    redir = _require_admin()
    if redir:
        return redir
    return send_file(str(_FRONTEND_DIR / "admin.html"))


@bp.route("/shunt-diagram/<path:filename>")
def shunt_diagram(filename):
    return send_file(str(_FRONTEND_DIR / "shunt_diagrams" / filename))