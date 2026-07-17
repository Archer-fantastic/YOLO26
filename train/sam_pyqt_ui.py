import sys

import cv2
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from sam import SAMInferencer


class SAMPyQtUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_sam()

    def init_ui(self):
        """Initialize the UI components."""
        # Set window properties
        self.setWindowTitle("SAM PyQt Interactive UI")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Top control bar
        control_bar = QHBoxLayout()

        # Upload button
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        control_bar.addWidget(self.upload_btn)

        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Points Mode", "Rectangle Mode"])
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        control_bar.addWidget(QLabel("Mode:"))
        control_bar.addWidget(self.mode_combo)

        # Run inference button
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.clicked.connect(self.run_inference)
        control_bar.addWidget(self.run_btn)

        # Clear button
        self.clear_btn = QPushButton("Clear Prompts")
        self.clear_btn.clicked.connect(self.clear_prompts)
        control_bar.addWidget(self.clear_btn)

        # Save button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        control_bar.addWidget(self.save_btn)

        main_layout.addLayout(control_bar)

        # Main split layout
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Image with prompts
        left_group = QGroupBox("Input Image & Prompts")
        left_layout = QVBoxLayout(left_group)

        self.image_label = QLabel("Upload an image to get started")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        left_layout.addWidget(self.image_label)

        # Right panel - Results
        right_group = QGroupBox("Segmentation Results")
        right_layout = QVBoxLayout(right_group)

        self.result_label = QLabel("Results will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        right_layout.addWidget(self.result_label)

        splitter.addWidget(left_group)
        splitter.addWidget(right_group)
        splitter.setSizes([600, 600])  # Equal width initially

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def init_sam(self):
        """Initialize the SAM model."""
        self.statusBar().showMessage("Loading SAM model...")
        self.inferencer = SAMInferencer()
        self.statusBar().showMessage("SAM model loaded successfully")

        # Initialize variables
        self.image = None
        self.pixmap = None
        self.temp_pixmap = None
        self.image_path = None

        # Mode: 0=points, 1=rectangle
        self.mode = 0

        # Points storage
        self.points = []
        self.labels = []

        # Rectangle storage
        self.rect_start = QPoint(-1, -1)
        self.rect_end = QPoint(-1, -1)
        self.drawing_rect = False

        # Colors
        self.positive_color = QColor(0, 255, 0)  # Green
        self.negative_color = QColor(255, 0, 0)  # Red
        self.rect_color = QColor(0, 255, 255)  # Cyan

    def upload_image(self):
        """Handle image upload."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)

            # Convert to QPixmap
            height, width, _channel = self.image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.pixmap = QPixmap.fromImage(q_img)

            # Scale to fit the label
            scaled_pixmap = self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.temp_pixmap = self.pixmap.copy()

            # Clear existing prompts
            self.clear_prompts()

            self.statusBar().showMessage(f"Image loaded: {file_path}")

    def change_mode(self, index):
        """Change between points and rectangle mode."""
        self.mode = index
        mode_name = "Points Mode" if index == 0 else "Rectangle Mode"
        self.statusBar().showMessage(f"Switched to {mode_name}")

    def mouse_press_event(self, event):
        """Handle mouse press events."""
        if not self.pixmap:
            return

        # Get mouse position relative to the image
        if self.image_label.pixmap():
            # Get the scaled pixmap displayed in the label
            scaled_pixmap = self.image_label.pixmap()

            # Calculate scaling factors: original image size / scaled display size
            scale_x = self.pixmap.width() / scaled_pixmap.width()
            scale_y = self.pixmap.height() / scaled_pixmap.height()

            # Get mouse position relative to the label
            label_pos = event.pos()

            # Adjust mouse position to original image coordinates
            x = int(label_pos.x() * scale_x)
            y = int(label_pos.y() * scale_y)

            if self.mode == 0:  # Points mode
                if event.button() == Qt.LeftButton:
                    # Add positive point
                    self.points.append([x, y])
                    self.labels.append(1)
                elif event.button() == Qt.RightButton:
                    # Add negative point
                    self.points.append([x, y])
                    self.labels.append(0)
                self.draw_points()

            elif self.mode == 1:  # Rectangle mode
                if event.button() == Qt.LeftButton:
                    # Start drawing rectangle
                    self.drawing_rect = True
                    self.rect_start = QPoint(x, y)
                    self.rect_end = QPoint(x, y)

    def mouse_move_event(self, event):
        """Handle mouse move events."""
        if not self.pixmap or not self.drawing_rect:
            return

        # Get mouse position relative to the image
        if self.image_label.pixmap():
            scaled_pixmap = self.image_label.pixmap()

            # Calculate scaling factors: original image size / scaled display size
            scale_x = self.pixmap.width() / scaled_pixmap.width()
            scale_y = self.pixmap.height() / scaled_pixmap.height()

            # Get mouse position relative to the label
            label_pos = event.pos()

            # Adjust mouse position to original image coordinates
            x = int(label_pos.x() * scale_x)
            y = int(label_pos.y() * scale_y)

        if self.mode == 1 and self.drawing_rect:
            # Update rectangle end point
            self.rect_end = QPoint(x, y)
            self.draw_rectangle()

    def mouse_release_event(self, event):
        """Handle mouse release events."""
        if not self.pixmap or not self.drawing_rect:
            return

        # Get mouse position relative to the image
        if self.image_label.pixmap():
            scaled_pixmap = self.image_label.pixmap()

            # Calculate scaling factors: original image size / scaled display size
            scale_x = self.pixmap.width() / scaled_pixmap.width()
            scale_y = self.pixmap.height() / scaled_pixmap.height()

            # Get mouse position relative to the label
            label_pos = event.pos()

            # Adjust mouse position to original image coordinates
            x = int(label_pos.x() * scale_x)
            y = int(label_pos.y() * scale_y)

        if self.mode == 1 and self.drawing_rect:
            # Finish drawing rectangle
            self.rect_end = QPoint(x, y)
            self.drawing_rect = False
            self.draw_rectangle()

    def draw_points(self):
        """Draw points on the image."""
        if not self.temp_pixmap:
            return

        # Create a copy of the original pixmap
        self.temp_pixmap = self.pixmap.copy()
        painter = QPainter(self.temp_pixmap)

        # Draw all points
        for i, (x, y) in enumerate(self.points):
            if self.labels[i] == 1:
                color = self.positive_color
                text = f"+{i}"
            else:
                color = self.negative_color
                text = f"-{i}"

            # Draw point
            pen = QPen(color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawEllipse(x - 5, y - 5, 10, 10)

            # Draw label
            pen = QPen(color, 1, Qt.SolidLine)
            painter.setPen(pen)
            painter.setFont(painter.font())
            painter.drawText(x + 10, y - 10, text)

        painter.end()

        # Update the label
        scaled_pixmap = self.temp_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def draw_rectangle(self):
        """Draw rectangle on the image."""
        if not self.temp_pixmap:
            return

        # Create a copy of the original pixmap
        self.temp_pixmap = self.pixmap.copy()
        painter = QPainter(self.temp_pixmap)

        # Draw rectangle
        pen = QPen(self.rect_color, 2, Qt.SolidLine)
        painter.setPen(pen)

        # Calculate rectangle coordinates
        x1 = min(self.rect_start.x(), self.rect_end.x())
        y1 = min(self.rect_start.y(), self.rect_end.y())
        x2 = max(self.rect_start.x(), self.rect_end.x())
        y2 = max(self.rect_start.y(), self.rect_end.y())

        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.end()

        # Update the label
        scaled_pixmap = self.temp_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def run_inference(self):
        """Run SAM inference."""
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please upload an image first!")
            return

        self.statusBar().showMessage("Running inference...")

        try:
            if self.mode == 0 and len(self.points) > 0:
                # Run with points
                results = self.inferencer.infer(self.image_path, points=self.points, labels=self.labels)
            elif self.mode == 1 and self.rect_start.x() != -1:
                # Run with rectangle
                # Calculate rectangle coordinates
                x1 = min(self.rect_start.x(), self.rect_end.x())
                y1 = min(self.rect_start.y(), self.rect_end.y())
                x2 = max(self.rect_start.x(), self.rect_end.x())
                y2 = max(self.rect_start.y(), self.rect_end.y())
                bboxes = [x1, y1, x2, y2]
                results = self.inferencer.infer(self.image_path, bboxes=bboxes)
            else:
                QMessageBox.warning(self, "Warning", "Please add some prompts first!")
                self.statusBar().showMessage("Ready")
                return

            # Display results
            self.display_results(results)
            self.statusBar().showMessage("Inference completed successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run inference: {e!s}")
            self.statusBar().showMessage(f"Error: {e!s}")

    def display_results(self, results):
        """Display the segmentation results."""
        # Convert results to QPixmap
        # First, get the result image
        result_img = results[0].plot()  # This returns a numpy array

        # Convert to QPixmap
        height, width, _channel = result_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        result_pixmap = QPixmap.fromImage(q_img)

        # Scale to fit the result label
        scaled_pixmap = result_pixmap.scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_label.setPixmap(scaled_pixmap)

        # Store for saving
        self.result_pixmap = result_pixmap

    def clear_prompts(self):
        """Clear all prompts."""
        self.points = []
        self.labels = []
        self.rect_start = QPoint(-1, -1)
        self.rect_end = QPoint(-1, -1)
        self.drawing_rect = False

        if self.pixmap:
            self.temp_pixmap = self.pixmap.copy()
            scaled_pixmap = self.temp_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

        self.statusBar().showMessage("All prompts cleared")

    def save_results(self):
        """Save the results."""
        if not hasattr(self, "result_pixmap") or not self.result_pixmap:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            if self.result_pixmap.save(file_path):
                self.statusBar().showMessage(f"Results saved to {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save results!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMPyQtUI()
    window.show()
    sys.exit(app.exec_())
