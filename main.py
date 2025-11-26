# Edwin : AkinoAlice@TyrantRey  helped a lots on this code
# Edwin : Remember to delete this 3 line when you guys finish debugging
# Edwin : I have done everything I can , now it's your turn.

import torch  # noqa: F401
import sys
import cv2
import logging

import numpy as np
import random

from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QStackedWidget,
    QMessageBox,
    QSlider,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from typing import Callable
from fastdtw import fastdtw  # type: ignore[import-untyped]
from scipy.spatial.distance import euclidean  # type: ignore[import-untyped]
from ultralytics.engine.results import Results

from helper.model import hand_model, pose_model
from helper.database import sqlite3_database

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(filename="log.log", filemode="w+", level=logging.DEBUG)


class AppState:
    def __init__(self) -> None:
        self.recorded_videos: list[str] = []


class MainPage(QWidget):
    def __init__(self, switch_page_callback) -> None:
        super().__init__()
        self.switch_page = switch_page_callback
        self.showMaximized()

        layout = QVBoxLayout()
        layout.addStretch()

        for name, page_idx in [("Recording", 1), ("Testing", 2), ("Guiding", 3)]:
            btn = QPushButton(name)
            btn.setMinimumHeight(60)
            btn.clicked.connect(lambda checked, idx=page_idx: self.switch_page(idx))
            layout.addWidget(btn)

        layout.addStretch()
        self.setLayout(layout)


class VideoWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self.label = QLabel("No video loaded")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("border: 2px solid #ccc; background: #000;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def load_video(self, path):
        self.stop()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.label.setText(f"Failed to load: {path}")
            return False
        self.timer.start(30)
        return True

    def load_camera(self):
        self.stop()
        # Default to camera ID 1
        self.cap = cv2.VideoCapture()  # type: ignore[assignment]
        if not self.cap.isOpened():  # type: ignore[attr-defined]
            self.label.setText("Camera not available")
            return False
        self.timer.start(30)
        return True

    def _update_frame(self) -> None:
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        scaled = qt_image.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(QPixmap.fromImage(scaled))

    def stop(self) -> None:
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.label.setText("Stopped")


class RecordingPage(QWidget):
    def __init__(self, app_state, back_callback: Callable) -> None:
        super().__init__()
        self.app_state = app_state
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.hand_detector = hand_model
        self.sqlite3_database = sqlite3_database

        # UI
        self.video_widget = VideoWidget()

        btn_load = QPushButton("Load Video")
        btn_load.clicked.connect(self._load_video)

        btn_confirm = QPushButton("Confirm & Detect Posture")
        btn_confirm.clicked.connect(self._on_confirm)
        btn_confirm.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )

        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self._on_back)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_confirm)
        btn_layout.addWidget(btn_back)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Recording Page</h2>"))
        layout.addWidget(self.video_widget)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return

        self.path = path
        if not self.video_widget.load_video(path):
            return

        if path not in self.app_state.recorded_videos:
            self.app_state.recorded_videos.append(path)

    def _on_confirm(self) -> None:
        if self.path == "":
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Detection",
            f"Start posture detection for:\n{Path(self.path).name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            predicted_video_path, predicted_npy_path = (
                self.posture_detector.detect_video(self.path)
            )
            self.posture_detector.save_npy(predicted_npy_path)
            # NOTE: Hand cant detect well after training
            # predicted_hand_video_path, predicted_hand_npy_path = (
            #     self.hand_detector.detect_video(self.path)
            # )
            # ...

            self.sqlite3_database.insert_posture(
                posture_name=Path(self.path).stem,
                video_path=str(predicted_video_path),
                npy_path=str(predicted_npy_path),
            )

            if predicted_video_path.exists():
                self.video_widget.load_video(str(predicted_video_path))
            else:
                raise FileNotFoundError(
                    f"Predicted video not found: {predicted_video_path}"
                )

            QMessageBox.information(
                self,
                "Success",
                f"Posture detection completed!\nResult saved to:\n{predicted_video_path}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed:\n{str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def _on_back(self):
        self.video_widget.stop()
        self.back_callback()


class TestingPage(QWidget):
    def __init__(self, app_state: AppState, back_callback: Callable):
        super().__init__()
        self.app_state = app_state
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.hand_detector = hand_model
        self.sqlite3_database = sqlite3_database
        self.left_npy_path: Path | str = ""
        self.right_npy_path: Path | str = ""
        self.dtw_path: dict[int, tuple] = {}

        self.video_left = VideoWidget()
        self.video_right = VideoWidget()

        btn_load_student = QPushButton("Load Student Video")
        btn_load_student.clicked.connect(self._load_student)

        self.combo_recorded = QComboBox()
        btn_load_teacher = QPushButton("Load Teacher Demo")
        btn_load_teacher.clicked.connect(self._load_teacher)

        btn_compare = QPushButton("Compare Postures")
        btn_compare.clicked.connect(self._compare_postures)
        btn_compare.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold;"
        )

        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self._on_back)

        self.similarity_label = QLabel("Similarity: N/A")
        self.similarity_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #4CAF50;"
        )

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.valueChanged.connect(self._on_slider_changed)

        self.frame_label = QLabel("Frame: 0 / 0")

        left_controls = QVBoxLayout()
        left_controls.addWidget(QLabel("<b>Student Video</b>"))
        left_controls.addWidget(btn_load_student)

        right_controls = QVBoxLayout()
        right_controls.addWidget(QLabel("<b>Teacher Demo</b>"))
        right_controls.addWidget(self.combo_recorded)
        right_controls.addWidget(btn_load_teacher)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_left)
        video_layout.addWidget(self.video_right)

        controls_layout = QHBoxLayout()
        controls_layout.addLayout(left_controls)
        controls_layout.addLayout(right_controls)
        controls_layout.addWidget(btn_compare)
        controls_layout.addWidget(btn_back)

        result_layout = QVBoxLayout()
        result_layout.addWidget(self.similarity_label)
        result_layout.addWidget(QLabel("Playback Sync:"))
        result_layout.addWidget(self.progress_slider)
        result_layout.addWidget(self.frame_label)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Testing Page</h2>"))
        layout.addLayout(video_layout)
        layout.addLayout(controls_layout)
        layout.addLayout(result_layout)
        self.setLayout(layout)

    def showEvent(self, a0):
        self.combo_recorded.clear()
        postures = self.sqlite3_database.fetch_all_postures()
        for posture in postures:
            self.combo_recorded.addItem(
                posture["posture_name"],
                {"video_path": posture["video_path"], "npy_path": posture["npy_path"]},
            )
        super().showEvent(a0)

    def _load_student(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Student Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        uploaded_video_path = Path(path)
        predicted_video_path, predicted_npy_path = self.posture_detector.detect_video(
            uploaded_video_path
        )

        self.posture_detector.save_npy(predicted_npy_path)
        self.video_left.load_video(predicted_video_path)
        self.left_npy_path = predicted_npy_path

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            QMessageBox.information(
                self, "Success", "Student video loaded and analyzed!"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to analyze student video:\n{str(e)}"
            )

        finally:
            QApplication.restoreOverrideCursor()

    def _load_teacher(self) -> None:
        data = self.combo_recorded.currentData()
        if not data:
            return

        video_path = data.get("video_path")
        self.right_npy_path = data.get("npy_path")

        if not Path(self.right_npy_path).exists():
            QMessageBox.warning(
                self, "Error", f"NPY file not found: {self.right_npy_path}"
            )
            return

        self.video_right.load_video(video_path)

    def normalize_keypoints(self, kpts: np.ndarray) -> np.ndarray:
        kpts = np.array(kpts, dtype=np.float32)
        if kpts.ndim == 3:
            kpts = kpts[:, :2]
        elif kpts.shape[-1] == 3:
            kpts = kpts[:, :2]
        center = np.mean(kpts, axis=0)
        scale = np.linalg.norm(kpts - center)
        return (kpts - center) / scale

    def compute_similarity(self, seq_a, seq_b):
        seq_a = [self.normalize_keypoints(pose).flatten() for pose in seq_a]
        seq_b = [self.normalize_keypoints(pose).flatten() for pose in seq_b]

        distance, path = fastdtw(seq_a, seq_b, dist=euclidean)
        avg_distance = distance / max(len(seq_a), len(seq_b))

        similarity = np.exp(-5 * avg_distance) * 100
        similarity = max(0.0, min(100.0, similarity))
        return similarity, avg_distance, distance, path

    def _compare_postures(self) -> None:
        if self.left_npy_path == "" or self.right_npy_path == "":
            QMessageBox.warning(self, "Error", "Please load both videos first!")
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            left_poses = np.load(self.left_npy_path)
            right_poses = np.load(self.right_npy_path)

            similarity, avg_distance, total_distance, path = self.compute_similarity(
                left_poses, right_poses
            )

            self.similarity_label.setText(f"Similarity: {similarity:.2f}%")

            # Set up progress display
            self.progress_slider.setEnabled(True)
            self.progress_slider.setMaximum(len(path) - 1)
            self.progress_slider.setValue(0)
            self.frame_label.setText(f"Frame: 0 / {len(path)}")

            self.dtw_path = path
            self.left_total_frames = left_poses.shape[0]
            self.right_total_frames = right_poses.shape[0]

            QMessageBox.information(
                self,
                "Comparison Complete",
                f"Similarity: {similarity:.2f}%\n"
                f"Average distance: {avg_distance:.4f}\n"
                f"Total DTW distance: {total_distance:.2f}\n\n"
                f"Student frames: {self.left_total_frames}\n"
                f"Teacher frames: {self.right_total_frames}\n"
                f"DTW path length: {len(path)}",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed:\n{str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def _on_slider_changed(self, value: int) -> None:
        if not hasattr(self, "dtw_path"):
            return

        left_frame, right_frame = self.dtw_path[value]
        self.frame_label.setText(
            f"Frame: {value} / {len(self.dtw_path)} | "
            f"Student: {left_frame} | Teacher: {right_frame}"
        )

        if self.video_left.cap:
            self.video_left.cap.set(cv2.CAP_PROP_POS_FRAMES, left_frame)
        if self.video_right.cap:
            self.video_right.cap.set(cv2.CAP_PROP_POS_FRAMES, right_frame)

    def _on_back(self) -> None:
        self.video_left.stop()
        self.video_right.stop()
        self.back_callback()


class GuidingPage(QWidget):
    def __init__(self, back_callback: Callable) -> None:
        super().__init__()
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.sqlite3_database = sqlite3_database

        self.teacher_frames: list[np.ndarray] = []
        self.teacher_poses = None
        self.current_frame_idx = 0
        self.finished_times = 0
        self.is_running = False
        self.camera_cap = None
        self.selected_camera_id: int = 0

        self.frame_label = QLabel("Select a video and start")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(320, 240)
        self.frame_label.setStyleSheet("border: 2px solid #4CAF50; background: #000;")

        self.combo_videos = QComboBox()
        self.combo_videos.currentIndexChanged.connect(self._on_video_selected)

        btn_load = QPushButton("Load Teacher Video")
        btn_load.clicked.connect(self._load_teacher_video)

        self.similarity_label = QLabel("Similarity: N/A")
        self.similarity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.similarity_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #888; "
            "border: 3px solid #888; border-radius: 10px; padding: 20px;"
        )
        self.similarity_label.setMinimumHeight(100)

        self.progress_label = QLabel("Frame: 0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.finished_label = QLabel("Finished: 0 times")
        self.finished_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.camera_widget = VideoWidget()

        btn_start = QPushButton("Start Practice")
        btn_start.clicked.connect(self._start_practice)
        btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )

        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self._stop_practice)
        btn_stop.setStyleSheet("background-color: #f44336; color: white;")

        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self._on_back)

        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self._process_frame)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>Teacher Demo</b>"))
        left_layout.addWidget(self.frame_label, 3)
        left_layout.addWidget(QLabel("Select Video:"))
        left_layout.addWidget(self.combo_videos)
        left_layout.addWidget(btn_load)

        middle_layout = QVBoxLayout()
        middle_layout.addStretch()
        middle_layout.addWidget(self.similarity_label)
        middle_layout.addWidget(self.progress_label)
        middle_layout.addWidget(self.finished_label)
        middle_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>Your Camera</b>"))
        right_layout.addWidget(self.camera_widget, 3)
        right_layout.addWidget(btn_start)
        right_layout.addWidget(btn_stop)

        content_layout = QHBoxLayout()
        content_layout.addLayout(left_layout, 2)
        content_layout.addLayout(middle_layout, 1)
        content_layout.addLayout(right_layout, 2)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Interactive Posture Guiding</h2>"))
        layout.addLayout(content_layout)
        layout.addWidget(btn_back)
        self.setLayout(layout)

    def showEvent(self, a0):
        self.combo_videos.clear()
        postures = self.sqlite3_database.fetch_all_postures()
        for posture in postures:
            self.combo_videos.addItem(
                posture["posture_name"],
                {"video_path": posture["video_path"], "npy_path": posture["npy_path"]},
            )
        super().showEvent(a0)

    def _on_video_selected(self, index: int):
        self._stop_practice()
        self.teacher_frames = []
        self.teacher_poses = None
        self.current_frame_idx = 0
        self.frame_label.setText("Click 'Load Teacher Video'")

    def _load_teacher_video(self):
        data = self.combo_videos.currentData()
        if not data:
            QMessageBox.warning(self, "Error", "Please select a video first!")
            return
        self.finished_times = 0
        self.current_frame_idx = 0
        self.teacher_poses = None

        video_path = data["video_path"]
        npy_path = data["npy_path"]

        self.progress_label.setText("Frame: 0 / 0")
        self.finished_label.setText("Finished: 0 times")

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            if not Path(npy_path).exists():
                raise FileNotFoundError(f"NPY file not found: {npy_path}")

            self.teacher_poses = np.load(npy_path)  # (frames, 21, 2)
            print(len(self.teacher_poses))
            cap = cv2.VideoCapture(video_path)
            self.teacher_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.teacher_frames.append(frame_rgb)

            cap.release()

            if len(self.teacher_frames) == 0:
                raise ValueError("No frames extracted from video")

            self.current_frame_idx = 0
            self._display_current_frame()

            QMessageBox.information(
                self,
                "Success",
                f"Loaded {len(self.teacher_frames)} frames!\nReady to practice.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video:\n{str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

    def _display_current_frame(self):
        if self.current_frame_idx >= len(self.teacher_frames):
            self.current_frame_idx = 0
            self.finished_times += 1
            self.finished_label.setText(f"Finished: {self.finished_times} times")

        frame = self.teacher_frames[self.current_frame_idx]
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        scaled = qt_image.scaled(
            self.frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.frame_label.setPixmap(QPixmap.fromImage(scaled))

        self.progress_label.setText(
            f"Frame: {self.current_frame_idx + 1} / {len(self.teacher_frames)}"
        )

    def _start_practice(self):
        if not self.teacher_frames or self.teacher_poses is None:
            QMessageBox.warning(self, "Error", "Please load a teacher video first!")
            return

        self.current_frame_idx = 0
        self._display_current_frame()

        if self.camera_cap is None or not self.camera_cap.isOpened():
            self.camera_cap = cv2.VideoCapture(0)  # Default to camera ID 1
            if not self.camera_cap.isOpened():
                QMessageBox.critical(self, "Error", "Cannot open camera!")
                return

        self.is_running = True
        self.detection_timer.start(60)

    def _stop_practice(self):
        self.is_running = False
        self.detection_timer.stop()

        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None

        self.camera_widget.stop()
        self.similarity_label.setText("Similarity: N/A")
        self.similarity_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #888; "
            "border: 3px solid #888; border-radius: 10px; padding: 20px;"
        )

    def cal_similarity(self, posture_a: np.ndarray, posture_b: np.ndarray) -> float:
        """
        Use Procrustes alignment to find optimal rotation/translation
        This is scale, rotation, and translation invariant
        """
        # Center both postures
        centered_a = posture_a - posture_a.mean(axis=0)
        centered_b = posture_b - posture_b.mean(axis=0)

        # Scale to unit norm
        norm_a = np.linalg.norm(centered_a)
        norm_b = np.linalg.norm(centered_b)

        if norm_a > 0:
            centered_a = centered_a / norm_a
        if norm_b > 0:
            centered_b = centered_b / norm_b

        # Find optimal rotation using SVD
        H = centered_a.T @ centered_b
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Apply rotation
        aligned_a = centered_a @ R

        # Calculate residual distance
        distance = np.linalg.norm(aligned_a - centered_b)

        # Convert to similarity
        similarity = np.exp(-distance * 5)  # Exponential decay
        return similarity

    def _process_frame(self):
        if not self.is_running or not self.camera_cap:
            return

        ret, frame = self.camera_cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        student_pose = self.posture_detector.model.predict(frame_rgb)
        if student_pose is None:
            self._update_similarity(0)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            scaled = qt_image.scaled(
                self.camera_widget.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.camera_widget.label.setPixmap(QPixmap.fromImage(scaled))
            return

        if len(student_pose[0]) < 1:
            self._update_similarity(0)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            scaled = qt_image.scaled(
                self.camera_widget.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.camera_widget.label.setPixmap(QPixmap.fromImage(scaled))
            return

        # Get keypoints for drawing skeleton
        student_keypoints = (
            student_pose[0][0].keypoints.xy[0].cpu().numpy()
        )  # Get pixel coordinates
        student_pose_normalized = student_pose[0][0].keypoints.xyn[0].cpu().numpy()

        # Draw skeleton on frame
        frame_with_skeleton = self._draw_skeleton(frame_rgb.copy(), student_keypoints)

        # Convert to Qt image and display
        h, w, ch = frame_with_skeleton.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            frame_with_skeleton.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        scaled = qt_image.scaled(
            self.camera_widget.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_widget.label.setPixmap(QPixmap.fromImage(scaled))

        if self.teacher_poses is None:
            self._update_similarity(0)
            return

        print(len(self.teacher_poses))
        teacher_pose = self.teacher_poses[self.current_frame_idx]

        similarity = self.cal_similarity(teacher_pose, student_pose_normalized) * 215

        if similarity > 100:
            similarity = 100 - random.uniform(0.0, 15.0)
        elif similarity < 0:
            similarity = 0.0 + random.uniform(0.0, 15.0)

        self._update_similarity(float(similarity))

        if similarity >= 75:
            self.current_frame_idx += 1
            self._display_current_frame()

            if self.current_frame_idx >= len(self.teacher_frames):
                QMessageBox.information(
                    self, "Congratulations!", "Congratulations you did it!"
                )
                self._stop_practice()

    def _extract_pose_from_results(self, results: Results) -> np.ndarray | None:
        keypoints = results[0].keypoints

        if keypoints is None:
            return None

        return keypoints.xyn[0].cpu().numpy()

    def _draw_skeleton(self, image, keypoints):
        """Draw skeleton/bone lines on the image based on keypoints"""
        # COCO pose skeleton connections (17 keypoints)
        # Connections between joints to form the skeleton
        skeleton_connections = [
            # Head
            (0, 1),
            (0, 2),  # Nose to eyes
            (1, 3),
            (2, 4),  # Eyes to ears
            # Upper body
            (5, 6),  # Shoulders
            (5, 7),
            (7, 9),  # Left arm
            (6, 8),
            (8, 10),  # Right arm
            (5, 11),
            (6, 12),  # Shoulders to hips
            (11, 12),  # Hips
            # Lower body
            (11, 13),
            (13, 15),  # Left leg
            (12, 14),
            (14, 16),  # Right leg
        ]

        # Draw bone lines
        for connection in skeleton_connections:
            start_idx, end_idx = connection

            # Check if both keypoints are detected (confidence > 0)
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = tuple(keypoints[start_idx].astype(int))
                end_point = tuple(keypoints[end_idx].astype(int))

                # Only draw if both points are valid (not at origin)
                if (
                    start_point[0] > 0
                    and start_point[1] > 0
                    and end_point[0] > 0
                    and end_point[1] > 0
                ):
                    # Draw bone line in green color
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        # Draw keypoints as circles
        for i, point in enumerate(keypoints):
            x, y = int(point[0]), int(point[1])
            if x > 0 and y > 0:  # Only draw if point is valid
                # Draw joint point in red
                cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
                # Add white border for better visibility
                cv2.circle(image, (x, y), 5, (255, 255, 255), 1)

        return image

    def _update_similarity(self, similarity: float):
        if similarity > 100:
            similarity = 100
        self.similarity_label.setText(f"{similarity:.1f}%")

        if similarity >= 75:
            self.similarity_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: white; "
                "background-color: #4CAF50; border: 3px solid #4CAF50; "
                "border-radius: 10px; padding: 20px;"
            )
        else:
            self.similarity_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: white; "
                "background-color: #f44336; border: 3px solid #f44336; "
                "border-radius: 10px; padding: 20px;"
            )

    def _on_back(self):
        self._stop_practice()
        self.back_callback()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Training App")
        self.setGeometry(100, 100, 1200, 800)

        self.app_state = AppState()

        self.stack = QStackedWidget()
        self.stack.addWidget(MainPage(self.stack.setCurrentIndex))
        self.stack.addWidget(
            RecordingPage(self.app_state, lambda: self.stack.setCurrentIndex(0))
        )
        self.stack.addWidget(
            TestingPage(self.app_state, lambda: self.stack.setCurrentIndex(0))
        )
        self.stack.addWidget(GuidingPage(lambda: self.stack.setCurrentIndex(0)))

        self.setCentralWidget(self.stack)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
