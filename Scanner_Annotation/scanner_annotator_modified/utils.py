import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


def cv_to_qt_pixmap(cv_img, width=None, height=None, from_rgb=False):
    """Convert OpenCV image to QPixmap for PySide6 display."""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if not from_rgb else cv_img
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h,
                      bytes_per_line, QImage.Format_RGB888)
    if width and height:
        return QPixmap.fromImage(qt_image).scaled(width, height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(qt_image)