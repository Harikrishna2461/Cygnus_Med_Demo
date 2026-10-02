import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import UltrasoundApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UltrasoundApp()
    window.show()
    sys.exit(app.exec())