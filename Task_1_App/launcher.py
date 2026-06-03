"""
CHIVA Clinical Assistant — desktop launcher.

Starts the Flask backend in a background thread and presents a PySide6
status window. The user double-clicks this (or the built .exe) to launch
the full app; their default browser opens automatically once the server
is ready.
"""

import os
import sys
import webbrowser
import logging
import traceback
from pathlib import Path

# ── Root resolution ───────────────────────────────────────────────────────────
# sys._MEIPASS  → set by PyInstaller (both --onefile and --onedir in v6+)
# sys.frozen    → True when running as a PyInstaller bundle
_IS_FROZEN = getattr(sys, "frozen", False)
_DATA_ROOT  = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
_WRITE_ROOT = Path(sys.executable).parent if _IS_FROZEN else _DATA_ROOT

_BACKEND = _DATA_ROOT / "backend"
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_DATA_ROOT))

# ── SSL certificate fix (PyInstaller bundles certifi but ssl module can't find ──
# the system CA store on the target machine; point it at the bundled cacert.pem)
if _IS_FROZEN:
    try:
        import certifi as _certifi
        _ca = _certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", _ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
        os.environ.setdefault("HTTPX_SSL_CERT_FILE", _ca)
    except Exception:
        pass

# ── Remove stale Qdrant lock file (left by crashed/killed previous run) ───────
for _lock in (_DATA_ROOT / "qdrant_storage" / ".lock",
              _WRITE_ROOT / "qdrant_storage" / ".lock"):
    try:
        _lock.unlink(missing_ok=True)
    except OSError:
        pass

# ── Patch config paths early (before any backend module imports them) ─────────
# config.py computes paths from Path(__file__).parent which resolves to
# _internal/ inside the bundle — a valid read-only dir for qdrant_storage,
# but we redirect db/log to the writable exe directory.
import config as _cfg
_cfg.QDRANT_PATH = str(_DATA_ROOT / "qdrant_storage")
_cfg.DB_PATH     = str(_WRITE_ROOT / "chat_history.db")
_cfg.LOG_FILE    = str(_WRITE_ROOT / "cmed_chat.log")

# ── PySide6 ───────────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QFrame,
    QSystemTrayIcon, QMenu,
)
from PySide6.QtGui import QIcon, QPixmap, QColor, QAction
from PySide6.QtCore import Qt, QTimer, QThread, Signal


# ── Log handler that forwards records to a Qt signal ─────────────────────────
class _GuiLogHandler(logging.Handler):
    def __init__(self, signal: Signal):
        super().__init__()
        self._sig = signal
        self.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord):
        try:
            self._sig.emit(self.format(record))
        except Exception:
            pass


# ── Flask server thread ───────────────────────────────────────────────────────
class _FlaskThread(QThread):
    log_line       = Signal(str)
    status_changed = Signal(str)   # "starting" | "running" | "error"

    def run(self):
        try:
            self.status_changed.emit("starting")

            # Wire our GUI handler before Flask logging kicks in
            root_log = logging.getLogger()
            root_log.setLevel(logging.INFO)
            root_log.addHandler(_GuiLogHandler(self.log_line))

            self.log_line.emit("Initialising services…")

            # Import Flask app — app.py calls services.init_services() at
            # module level, so do NOT call it here too (double Qdrant open = lock conflict)
            import app as _app_mod

            # Initialise DB + BM25 indexes
            _app_mod._startup()

            from config import HOST, PORT

            self.status_changed.emit("running")
            self.log_line.emit(f"Server ready → http://{HOST}:{PORT}")

            _app_mod.app.run(
                host=HOST,
                port=PORT,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        except Exception as exc:
            self.status_changed.emit("error")
            self.log_line.emit(f"FATAL: {exc}")
            self.log_line.emit(traceback.format_exc())


# ── Coloured status indicator ─────────────────────────────────────────────────
class _Dot(QLabel):
    _CLR = {"starting": "#f59e0b", "running": "#22c55e", "error": "#ef4444"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(11, 11)
        self._set("starting")

    def _set(self, state: str):
        c = self._CLR.get(state, "#94a3b8")
        self.setStyleSheet(f"background:{c}; border-radius:5px;")

    def update_state(self, state: str):
        self._set(state)


# ── Stylesheet ────────────────────────────────────────────────────────────────
_QSS = """
QMainWindow { background: #0f172a; }
QWidget#root { background: #0f172a; }

QLabel#title  { color: #ffffff; font-size: 20px; font-weight: 700; }
QLabel#sub    { color: #94a3b8; font-size: 12px; }
QLabel#stat   { color: #94a3b8; font-size: 12px; }
QLabel#logttl { color: #475569; font-size: 11px; }

QPushButton#open {
    background: #2563eb; color: #ffffff;
    border: none; border-radius: 8px;
    font-size: 14px; font-weight: 600;
    padding: 11px 28px;
}
QPushButton#open:hover    { background: #1d4ed8; }
QPushButton#open:disabled { background: #1e293b; color: #475569; }

QTextEdit#log {
    background: #1e293b; color: #94a3b8;
    border: 1px solid #334155; border-radius: 6px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
    selection-background-color: #334155;
}

QScrollBar:vertical { background: #1e293b; width: 6px; border-radius: 3px; }
QScrollBar::handle:vertical { background: #334155; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
"""


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, thread: _FlaskThread):
        super().__init__()
        self._thread = thread
        self._host: str = "127.0.0.1"
        self._port: int = 7860

        self.setWindowTitle("CHIVA Clinical Assistant")
        self.setMinimumSize(600, 480)
        self.setStyleSheet(_QSS)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(44, 38, 44, 38)
        lay.setSpacing(18)

        # ── Header
        title = QLabel("CHIVA Clinical Assistant")
        title.setObjectName("title")
        sub = QLabel("AI-powered venous shunt classification & ligation planning")
        sub.setObjectName("sub")
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(4)

        # ── Status row
        row = QHBoxLayout()
        row.setSpacing(8)
        self._dot = _Dot()
        self._stat_lbl = QLabel("Starting server…")
        self._stat_lbl.setObjectName("stat")
        row.addWidget(self._dot)
        row.addWidget(self._stat_lbl)
        row.addStretch()
        lay.addLayout(row)

        # ── Open button
        self._btn = QPushButton("Open in Browser")
        self._btn.setObjectName("open")
        self._btn.setEnabled(False)
        self._btn.clicked.connect(self._open)
        lay.addWidget(self._btn, alignment=Qt.AlignLeft)

        # ── Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #1e293b;")
        lay.addWidget(sep)

        # ── Log
        lbl = QLabel("Server log")
        lbl.setObjectName("logttl")
        lay.addWidget(lbl)
        self._log = QTextEdit()
        self._log.setObjectName("log")
        self._log.setReadOnly(True)
        lay.addWidget(self._log)

        # ── Connect signals
        thread.log_line.connect(self._append)
        thread.status_changed.connect(self._on_status)

        # ── Tray
        self._tray = self._make_tray()

    # ── Tray setup ────────────────────────────────────────────────────────────
    def _make_tray(self) -> QSystemTrayIcon:
        pix = QPixmap(16, 16)
        pix.fill(QColor("#2563eb"))
        tray = QSystemTrayIcon(QIcon(pix), self)

        menu = QMenu()
        a_open  = QAction("Open App",      self); a_open.triggered.connect(self._open)
        a_show  = QAction("Show Window",   self); a_show.triggered.connect(self.show)
        a_quit  = QAction("Quit",          self); a_quit.triggered.connect(self._quit)
        menu.addAction(a_open)
        menu.addAction(a_show)
        menu.addSeparator()
        menu.addAction(a_quit)

        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda r: self.show() if r == QSystemTrayIcon.DoubleClick else None
        )
        tray.show()
        return tray

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _append(self, text: str):
        self._log.append(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_status(self, state: str):
        self._dot.update_state(state)
        if state == "starting":
            self._stat_lbl.setText("Starting server…")
            self._btn.setEnabled(False)
        elif state == "running":
            from config import HOST, PORT
            self._host, self._port = HOST, PORT
            self._stat_lbl.setText(f"Running  ·  http://{HOST}:{PORT}")
            self._btn.setEnabled(True)
            QTimer.singleShot(600, self._open)   # auto-open once
        elif state == "error":
            self._stat_lbl.setText("Server error — see log below")
            self._btn.setEnabled(False)

    def _open(self):
        webbrowser.open(f"http://{self._host}:{self._port}")

    # ── Window X button → hard exit (releases Qdrant lock immediately) ──────────
    def closeEvent(self, event):
        self._quit()

    def _quit(self):
        self._tray.hide()
        import os
        os._exit(0)   # hard kill — releases all file locks including Qdrant's


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # Ensure writable data directory exists next to the exe
    _WRITE_ROOT.mkdir(parents=True, exist_ok=True)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("CHIVA Clinical Assistant")
    qt_app.setQuitOnLastWindowClosed(False)

    thread = _FlaskThread()
    window = MainWindow(thread)
    window.show()
    thread.start()

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
