
from __future__ import annotations

import sys
import time
from collections import Counter

import cv2
import mediapipe as mp
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout,
    QWidget
)


# ============================================================
# VISION-X AI — CONFIGURATION
# ============================================================

WINDOW_TITLE = "VISION-X AI | Visual Intelligence Console"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
OBJECT_MODEL = "yolov8n.pt"
OBJECT_CONFIDENCE = 0.35
OBJECT_IMAGE_SIZE = 416

# Facial-expression analysis is optional and CPU-intensive.
EMOTION_INTERVAL_SECONDS = 2.0


# ============================================================
# HAND ANALYSIS
# ============================================================

class HandAnalyzer:
    """Hand landmarks, approximate finger count, and basic gestures."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils

        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )

    @staticmethod
    def count_fingers(landmarks, handedness):
        """
        Approximate raised-finger count.

        Four fingers use fingertip/PIP vertical positions.
        Thumb extension uses horizontal landmark positions.
        Perspective, hand rotation, and mirrored views can affect
        the result, so this is a baseline rather than a universal rule.
        """
        points = landmarks.landmark
        count = 0

        # Index, middle, ring, little finger.
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if points[tip].y < points[pip].y:
                count += 1

        # Basic thumb-extension heuristic.
        thumb_tip = points[4]
        thumb_ip = points[3]

        if handedness == "Right":
            if thumb_tip.x < thumb_ip.x:
                count += 1
        else:
            if thumb_tip.x > thumb_ip.x:
                count += 1

        return count

    @staticmethod
    def classify_gesture(landmarks, finger_count):
        """Small rule-based vocabulary of static hand gestures."""
        points = landmarks.landmark

        if finger_count == 0:
            return "FIST"

        if finger_count == 5:
            return "OPEN PALM"

        # Approximate thumbs-up: thumb high, other fingertips folded.
        thumb_extended = points[4].y < points[3].y
        other_fingers_folded = all(
            points[tip].y > points[pip].y
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )

        if thumb_extended and other_fingers_folded:
            return "THUMBS UP"

        if finger_count == 1:
            return "ONE FINGER"
        if finger_count == 2:
            return "TWO FINGERS"
        if finger_count == 3:
            return "THREE FINGERS"
        if finger_count == 4:
            return "FOUR FINGERS"

        return "UNKNOWN"

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        hand_summaries = []

        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []

            for i, hand_landmarks in enumerate(
                results.multi_hand_landmarks
            ):
                hand_label = "Unknown"

                if i < len(handedness_list):
                    hand_label = (
                        handedness_list[i].classification[0].label
                    )

                fingers = self.count_fingers(
                    hand_landmarks, hand_label
                )
                gesture = self.classify_gesture(
                    hand_landmarks, fingers
                )

                self.drawer.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

                wrist = hand_landmarks.landmark[0]
                height, width = frame.shape[:2]
                x = int(wrist.x * width)
                y = int(wrist.y * height)

                cv2.putText(
                    frame,
                    f"{hand_label}: {fingers} fingers | {gesture}",
                    (max(8, x - 40), max(25, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (70, 240, 150),
                    2,
                    cv2.LINE_AA,
                )

                hand_summaries.append({
                    "hand": hand_label,
                    "fingers": fingers,
                    "gesture": gesture,
                })

        return frame, hand_summaries

    def close(self):
        self.detector.close()


# ============================================================
# BODY POSE ANALYSIS
# ============================================================

class PoseAnalyzer:
    """Draw approximate body landmarks and upper-body connections."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.drawer = mp.solutions.drawing_utils

        self.detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if results.pose_landmarks:
            self.drawer.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        return frame, results.pose_landmarks is not None

    def close(self):
        self.detector.close()


# ============================================================
# OPTIONAL FACIAL-EXPRESSION ANALYSIS
# ============================================================

class EmotionAnalyzer:
    """
    Optional DeepFace integration.

    Output represents a model-estimated facial expression, not a
    verified internal emotion. Run inference infrequently on CPU.
    """

    def __init__(self):
        from deepface import DeepFace
        self.DeepFace = DeepFace

    def analyze(self, frame):
        try:
            result = self.DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, dict):
                result = [result]

            output = []
            for item in result:
                emotion = item.get("dominant_emotion", "unknown")
                region = item.get("region", {}) or {}

                output.append((
                    emotion,
                    int(region.get("x", 0)),
                    int(region.get("y", 0)),
                    int(region.get("w", 0)),
                    int(region.get("h", 0)),
                ))

            return output

        except Exception:
            return []

    @staticmethod
    def draw(frame, predictions):
        for emotion, x, y, w, h in predictions:
            if w > 0 and h > 0:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (255, 180, 60), 2
                )
                cv2.putText(
                    frame,
                    f"Expression estimate: {emotion}",
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 180, 60),
                    2,
                    cv2.LINE_AA,
                )
        return frame


# ============================================================
# BACKGROUND CAMERA AND INFERENCE WORKER
# ============================================================

class VisionWorker(QThread):
    frame_ready = Signal(object, dict)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        camera_index=0,
        enable_objects=True,
        enable_pose=True,
        enable_emotion=False,
    ):
        super().__init__()
        self.camera_index = camera_index
        self.enable_objects = enable_objects
        self.enable_pose = enable_pose
        self.enable_emotion = enable_emotion
        self.running = True

    def stop(self):
        self.running = False
        self.wait(5000)

    def run(self):
        camera = None
        hands = None
        pose = None

        try:
            camera = cv2.VideoCapture(
                self.camera_index, cv2.CAP_DSHOW
            )

            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(self.camera_index)

            if not camera.isOpened():
                self.error_occurred.emit(
                    f"Cannot open camera {self.camera_index}. "
                    "Check Windows camera permissions or try index 1."
                )
                return

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            hands = HandAnalyzer()
            if self.enable_pose:
                pose = PoseAnalyzer()

            object_model = None
            object_error = False
            emotion_model = None
            emotion_error = False
            last_emotion_run = 0.0
            cached_emotions = []

            frame_counter = 0
            fps = 0.0
            fps_start = time.perf_counter()

            self.status_changed.emit("CAMERA CONNECTED")

            while self.running:
                ok, frame = camera.read()

                if not ok:
                    self.msleep(20)
                    continue

                frame = cv2.flip(frame, 1)

                # 1. Hand landmarks, finger counts, and gestures.
                frame, hand_data = hands.process(frame)

                # 2. Body pose estimation.
                pose_detected = False
                if pose is not None:
                    frame, pose_detected = pose.process(frame)

                # 3. General-purpose object detection.
                detected_objects = []

                if self.enable_objects:
                    try:
                        if object_model is None:
                            self.status_changed.emit(
                                "LOADING YOLO MODEL..."
                            )
                            from ultralytics import YOLO
                            object_model = YOLO(OBJECT_MODEL)
                            self.status_changed.emit("YOLO READY")

                        result = object_model.predict(
                            source=frame,
                            imgsz=OBJECT_IMAGE_SIZE,
                            conf=OBJECT_CONFIDENCE,
                            device="cpu",
                            verbose=False,
                        )[0]

                        if result.boxes is not None:
                            names = result.names

                            for box in result.boxes:
                                class_id = int(box.cls[0].item())
                                confidence = float(
                                    box.conf[0].item()
                                )
                                label = names.get(
                                    class_id, str(class_id)
                                )
                                x1, y1, x2, y2 = map(
                                    int, box.xyxy[0].tolist()
                                )

                                cv2.rectangle(
                                    frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (60, 205, 255),
                                    2,
                                )
                                cv2.putText(
                                    frame,
                                    f"{label} {confidence:.0%}",
                                    (x1, max(20, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (60, 205, 255),
                                    2,
                                    cv2.LINE_AA,
                                )
                                detected_objects.append(label)

                    except Exception as exc:
                        if not object_error:
                            self.status_changed.emit(
                                f"OBJECT DETECTION ERROR: {exc}"
                            )
                            object_error = True

                # 4. Optional facial-expression inference.
                if self.enable_emotion:
                    now = time.perf_counter()

                    if now - last_emotion_run >= EMOTION_INTERVAL_SECONDS:
                        try:
                            if emotion_model is None:
                                emotion_model = EmotionAnalyzer()

                            cached_emotions = emotion_model.analyze(
                                frame.copy()
                            )
                            last_emotion_run = now

                        except Exception as exc:
                            if not emotion_error:
                                self.status_changed.emit(
                                    "EXPRESSION MODULE ERROR: "
                                    f"{exc}. Install deepface if needed."
                                )
                                emotion_error = True

                    frame = EmotionAnalyzer.draw(
                        frame, cached_emotions
                    )

                # 5. FPS measurement.
                frame_counter += 1
                now = time.perf_counter()
                elapsed = now - fps_start

                if elapsed >= 1.0:
                    fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_start = now

                cv2.rectangle(
                    frame, (8, 8), (245, 42), (18, 28, 42), -1
                )
                cv2.putText(
                    frame,
                    f"VISION-X AI | {fps:.1f} FPS",
                    (17, 31),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (80, 245, 175),
                    2,
                    cv2.LINE_AA,
                )

                statistics = {
                    "fps": round(fps, 1),
                    "hands": hand_data,
                    "objects": sorted(set(detected_objects)),
                    "pose": pose_detected,
                    "emotion_enabled": self.enable_emotion,
                }

                self.frame_ready.emit(frame, statistics)
                self.msleep(1)

        except Exception as exc:
            self.error_occurred.emit(str(exc))

        finally:
            if hands is not None:
                hands.close()
            if pose is not None:
                pose.close()
            if camera is not None:
                camera.release()

            self.status_changed.emit("CAMERA STOPPED")


# ============================================================
# DESKTOP DASHBOARD
# ============================================================

class VisionXWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1250, 780)
        self.setMinimumSize(950, 620)
        self.worker = None

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0b1020;
                color: #e8eef8;
            }
            QFrame#Panel {
                background: #111a2d;
                border: 1px solid #293953;
                border-radius: 12px;
            }
            QLabel#Muted { color: #93a5be; }
            QLabel#Metric {
                color: #63efbb;
                font-size: 17px;
                font-weight: bold;
            }
            QPushButton {
                background: #1a2942;
                border: 1px solid #354966;
                border-radius: 7px;
                padding: 9px;
            }
            QPushButton:hover { background: #263c5b; }
            QPushButton#Start {
                background: #087c64;
                border: 1px solid #19b994;
                font-weight: bold;
            }
            QComboBox {
                background: #111a2d;
                border: 1px solid #354966;
                padding: 7px;
                border-radius: 6px;
            }
        """)

        self.build_ui()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        header = QHBoxLayout()
        brand = QVBoxLayout()

        title = QLabel("VISION-X AI")
        title.setFont(QFont("Segoe UI", 23, QFont.Bold))

        subtitle = QLabel(
            "MULTIMODAL VISUAL INTELLIGENCE | CPU EDITION"
        )
        subtitle.setObjectName("Muted")

        brand.addWidget(title)
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()

        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setObjectName("Muted")
        header.addWidget(self.status_label)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        camera_panel = QFrame()
        camera_panel.setObjectName("Panel")
        camera_layout = QVBoxLayout(camera_panel)

        camera_title = QLabel("LIVE CAMERA FEED")
        camera_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        camera_layout.addWidget(camera_title)

        self.video_label = QLabel(
            "CAMERA OFFLINE\n\nPress START CAMERA to begin"
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(550, 350)
        self.video_label.setStyleSheet(
            "background:#060b14; border-radius:8px; color:#93a5be;"
        )
        camera_layout.addWidget(self.video_label, 1)

        local_note = QLabel(
            "Local processing • No video recording by default"
        )
        local_note.setObjectName("Muted")
        camera_layout.addWidget(local_note)
        body.addWidget(camera_panel, 3)

        side_panel = QFrame()
        side_panel.setObjectName("Panel")
        side = QVBoxLayout(side_panel)

        heading = QLabel("PERCEPTION ANALYTICS")
        heading.setFont(QFont("Segoe UI", 11, QFont.Bold))
        side.addWidget(heading)

        self.fps_value = self.add_metric(side, "PROCESSING FPS", "—")
        self.hand_value = self.add_metric(side, "HANDS / FINGERS", "None")
        self.gesture_value = self.add_metric(side, "GESTURES", "—")
        self.object_value = self.add_metric(side, "OBJECTS", "—")
        self.pose_value = self.add_metric(side, "BODY POSE", "—")
        self.emotion_value = self.add_metric(
            side, "FACIAL EXPRESSION", "Optional"
        )

        side.addWidget(QLabel("CAMERA INDEX"))
        self.camera_index = QComboBox()
        self.camera_index.addItems(["0", "1", "2"])
        side.addWidget(self.camera_index)

        self.objects_check = QCheckBox("Enable YOLO object detection")
        self.objects_check.setChecked(True)
        side.addWidget(self.objects_check)

        self.pose_check = QCheckBox("Enable body-pose estimation")
        self.pose_check.setChecked(True)
        side.addWidget(self.pose_check)

        self.emotion_check = QCheckBox(
            "Enable facial-expression estimates"
        )
        self.emotion_check.setChecked(False)
        side.addWidget(self.emotion_check)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("START CAMERA")
        self.start_button.setObjectName("Start")
        self.stop_button = QPushButton("STOP")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        side.addLayout(buttons)

        disclaimer = QLabel(
            "Facial-expression categories are uncertain model "
            "estimates, not verified emotions."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setObjectName("Muted")
        side.addWidget(disclaimer)

        body.addWidget(side_panel, 1)

        footer = QLabel(
            "VISION-X AI | COMPUTER VISION LAB | LOCAL INFERENCE"
        )
        footer.setObjectName("Muted")
        root.addWidget(footer)

    @staticmethod
    def add_metric(layout, name, initial_value):
        frame = QFrame()
        frame.setStyleSheet(
            "background:#0b1324; border-radius:7px; padding:5px;"
        )
        box = QVBoxLayout(frame)

        label = QLabel(name)
        label.setObjectName("Muted")

        value = QLabel(initial_value)
        value.setObjectName("Metric")
        value.setWordWrap(True)

        box.addWidget(label)
        box.addWidget(value)
        layout.addWidget(frame)

        return value

    def start_camera(self):
        if self.worker is not None:
            return

        self.start_button.setEnabled(False)
        self.status_label.setText("STARTING...")

        self.worker = VisionWorker(
            camera_index=int(self.camera_index.currentText()),
            enable_objects=self.objects_check.isChecked(),
            enable_pose=self.pose_check.isChecked(),
            enable_emotion=self.emotion_check.isChecked(),
        )

        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_changed.connect(
            self.status_label.setText
        )
        self.worker.error_occurred.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)

        self.worker.start()
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        worker = self.worker

        if worker is not None:
            worker.stop()
            self.worker = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(
            "CAMERA OFFLINE\n\nPress START CAMERA to begin"
        )

    def update_frame(self, frame, stats):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

        self.fps_value.setText(str(stats["fps"]))

        hands = stats["hands"]
        if hands:
            self.hand_value.setText(
                " | ".join(
                    f'{h["hand"]}: {h["fingers"]}'
                    for h in hands
                )
            )
            self.gesture_value.setText(
                " | ".join(h["gesture"] for h in hands)
            )
        else:
            self.hand_value.setText("No hands detected")
            self.gesture_value.setText("—")

        objects = stats["objects"]
        self.object_value.setText(
            ", ".join(objects[:7]) if objects else "None detected"
        )

        self.pose_value.setText(
            "Detected" if stats["pose"] else "Not detected"
        )

        self.emotion_value.setText(
            "Enabled" if stats["emotion_enabled"] else "Disabled"
        )

    def show_error(self, message):
        self.status_label.setText("ERROR")
        QMessageBox.critical(self, "VISION-X AI", message)

    def worker_finished(self):
        # Do not forcibly destroy a running QThread.
        if self.worker is not None and not self.worker.isRunning():
            self.worker = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = VisionXWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
