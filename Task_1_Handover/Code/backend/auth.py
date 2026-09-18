from functools import wraps
from flask import session, jsonify


def login_required(f):
    """Decorator: returns a 401 JSON error if no valid session cookie is present,
    otherwise calls the wrapped route as normal. Usage: applied to every protected
    /api/* route across routes/clinical.py, general.py, sessions.py, feedback.py
    (API-level auth — the page-level equivalent for HTML routes is
    routes/views.py's _require_login())."""
    @wraps(f)
    def decorated(*args, **kwargs):
        """The actual wrapper Flask calls in place of the decorated route — checks the
        session, then delegates to the original function f() if authenticated."""
        if "user_id" not in session:
            return jsonify({"error": "Authentication required", "redirect": "/login"}), 401
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator: returns 401 if not logged in, 403 if logged in but not an admin
    (session["is_admin"] is falsy), otherwise calls the wrapped route. Usage: applied
    to every /api/admin/* route in routes/admin.py. Unlike routes/views.py's
    _require_admin() (used for the /admin HTML page), this only checks is_admin — it
    has no extra username whitelist."""
    @wraps(f)
    def decorated(*args, **kwargs):
        """The actual wrapper Flask calls in place of the decorated route — checks the
        session and admin flag, then delegates to the original function f() if
        authorised."""
        if "user_id" not in session:
            return jsonify({"error": "Authentication required", "redirect": "/login"}), 401
        if not session.get("is_admin"):
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated
