from .views import bp as views_bp
from .status import bp as status_bp
from .sessions import bp as sessions_bp
from .clinical import bp as clinical_bp
from .general import bp as general_bp
from .feedback import bp as feedback_bp
from .auth import bp as auth_bp
from .admin import bp as admin_bp

__all__ = [
    "views_bp", "status_bp", "sessions_bp",
    "clinical_bp", "general_bp", "feedback_bp",
    "auth_bp", "admin_bp",
]