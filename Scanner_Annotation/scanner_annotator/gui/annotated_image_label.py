from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPainter, QColor, QFont, QMouseEvent, QWheelEvent
from utils import cv_to_qt_pixmap


class AnnotatedImageLabel(QLabel):
    # Signal emitted with image coordinates when the label is clicked
    clicked = Signal(int, int, QMouseEvent)  # x, y coordinates of the click
    scrolled = Signal(bool, QWheelEvent)

    def __init__(self, parent=None, interactive=True):
        super().__init__(parent)
        self.setScaledContents(True)
        self.pixmap_image = None  # Stores the QPixmap version of the image
        self.interactive = interactive
        # List of tuples: (x, y, instance_id, is_positive)
        self.annotation_marks = []
        self.crop_mode = False
        self.temp_crop = (None, None)  # Temporary crop points (pt1, pt2)
        self.crop_region = None  # Finalized crop region (xmin, ymin, xmax, ymax)
        self.current_vein_id = None
        self.is_positive = True
        
        # The follower label
        self.setMouseTracking(True)
        self.follower = QLabel("ID: ", self)
        self.follower.setStyleSheet("background-color: black; color: white; padding: 2px; font-size: 16px;")
        self.follower.hide()

    def add_annotation_mark(self, x, y, instance, is_positive):
        self.annotation_marks.append((x, y, instance, is_positive))
        self.update()

    def clear_annotation_marks(self):
        self.annotation_marks = []
        self.update()

    def set_crop_mode(self, enabled):
        self.crop_mode = enabled
        self.temp_crop = (None, None)
        self.update()

    def set_temp_crop(self, pt1, pt2):
        self.temp_crop = (pt1, pt2)
        self.update()

    def set_crop_region(self, region):
        self.crop_region = region
        self.update()

    def mousePressEvent(self, event):
        # Ignore clicks when not interactive, no image, or not left mouse
        if not self.interactive or not self.pixmap_image: # or event.button() != Qt.LeftButton:
            return
        label_size = self.size()
        pixmap_size = self.pixmap_image.size()
        # Compute ratios to map widget coordinates back to image coordinates
        x_ratio = pixmap_size.width() / label_size.width()
        y_ratio = pixmap_size.height() / label_size.height()
        x = int(event.position().x() * x_ratio)
        y = int(event.position().y() * y_ratio)

        print(f"Clicked at image coordinates: ({x}, {y})")
        self.clicked.emit(x, y, event)
        self.update()
    
    def wheelEvent(self, event):
        # Coordinates relative to the widget
        x = int(event.position().x())
        y = int(event.position().y())
        
        # Scroll distance/direction (usually multiples of 120)
        delta = event.angleDelta().y()
        
        if delta > 0:
            is_up = True
        else:
            is_up = False
        
        # Emit the signal
        self.scrolled.emit(is_up, event)
        
        # Accept the event to stop it from bubbling up to parent widgets
        event.accept()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap_image:
            return
        # Draw annotation marks
        if self.annotation_marks:
            painter = QPainter(self)
            font = QFont()
            font.setBold(True)
            font.setPointSize(14)
            painter.setFont(font)
            # Calculate the scaling factors between pixmap image and label (widget)
            label_size = self.size()
            pixmap_size = self.pixmap_image.size()
            x_ratio = label_size.width() / pixmap_size.width()
            y_ratio = label_size.height() / pixmap_size.height()
            for x, y, instance, is_positive in self.annotation_marks:
                # Map image coordinates to label coordinates for drawing
                draw_x = int(x * x_ratio)
                draw_y = int(y * y_ratio)
                # Green for positive, Red for negative
                color = QColor(0, 200, 0) if is_positive else QColor(200, 0, 0)
                painter.setPen(color)
                painter.drawText(draw_x, draw_y, str(instance))
            painter.end()

        # Draw crop region (and temp crop cross)
        painter = QPainter(self)
        label_size = self.size()
        pixmap_size = self.pixmap_image.size()
        # Ratios to map image coords to widget coords
        x_ratio = label_size.width() / pixmap_size.width()
        y_ratio = label_size.height() / pixmap_size.height()
        # Draw existing crop region if any
        if getattr(self, 'crop_region', None) is not None:
            xmin, ymin, xmax, ymax = self.crop_region
            draw_x0 = int(xmin * x_ratio)
            draw_x1 = int(xmax * x_ratio)
            draw_y0 = int(ymin * y_ratio)
            draw_y1 = int(ymax * y_ratio)
            painter.setPen(QColor(255, 0, 0)) # Red color for crop region
            painter.drawRect(draw_x0, draw_y0, draw_x1 - draw_x0, draw_y1 - draw_y0)
        # Draw cross if first point of temp crop is set
        if self.temp_crop[0] is not None:
            pt1 = self.temp_crop[0]
            draw_x0 = int(pt1[0] * x_ratio)
            draw_y0 = int(pt1[1] * y_ratio)
            painter.setPen(QColor(255, 255, 0)) # Yellow color for temp crop
            # Draw a '+' cross at (draw_x0, draw_y0)
            painter.drawLine(draw_x0 - 6, draw_y0, draw_x0 + 6, draw_y0)
            painter.drawLine(draw_x0, draw_y0 - 6, draw_x0, draw_y0 + 6)
        painter.end()

    def enterEvent(self, event):
        if self.current_vein_id is not None:
            self.follower.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.follower.hide()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        # Position follower at mouse coordinates with an offset
        self.follower.move(event.position().toPoint() + QPoint(10, 10))
        super().mouseMoveEvent(event)
        
    def set_active_label(self, label_id, is_positive):
        self.current_vein_id = label_id
        self.is_positive = is_positive
        self.follower.setText(f"ID: {int(self.current_vein_id)}")
        if is_positive:
            self.follower.setStyleSheet("""
                color: lime;
                background-color: black;
                font-size: 16px;
            """)
        else:
            self.follower.setStyleSheet("""
                color: red;
                background-color: black;
                font-size: 16px;
            """)