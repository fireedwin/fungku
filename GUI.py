import torch  # noqa: F401
import sys
import logging
import numpy as np
import random
import time
from pathlib import Path
from typing import Callable
from collections import deque

# PyQt5 Imports
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

# 3rd Party Imports
import cv2
from fastdtw import fastdtw  # type: ignore[import-untyped]
from scipy.spatial.distance import euclidean  # type: ignore[import-untyped]
from ultralytics.engine.results import Results

# Local Imports
# Ensure you have the helper folder with model.py and database.py in the same directory
from helper.model import pose_model
from helper.database import sqlite3_database

# Logging Setup
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(filename="log.log", filemode="w+", level=logging.DEBUG)


def get_working_cameras(max_check=3):
    """
    Scans indices 0 to max_check.
    Tries BOTH DirectShow and Default backends.
    """
    working_cams = []
    print("Scanning for cameras...")
    for i in range(max_check):
        # 1. Try DirectShow (Best for OBS / Virtual Cams)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  - Camera {i}: OK (DirectShow)")
                working_cams.append((i, cv2.CAP_DSHOW, f"Camera {i} (DSHOW)"))
                cap.release()
                continue 
        cap.release()

        # 2. Try Default/MSMF (Best for built-in Laptop Cams)
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  - Camera {i}: OK (Default)")
                working_cams.append((i, cv2.CAP_ANY, f"Camera {i} (Default)"))
                cap.release()
                continue
        cap.release()
    
    if not working_cams:
        print("  - No cameras responded. Defaulting to Camera 0.")
        working_cams.append((0, cv2.CAP_ANY, "Camera 0 (Fallback)"))
        
    return working_cams


class AppState:
    def __init__(self) -> None:
        self.recorded_videos: list[str] = []
        self.camera_config = (0, cv2.CAP_ANY)


class MainPage(QWidget):
    def __init__(self, switch_page_callback) -> None:
        super().__init__()
        self.switch_page = switch_page_callback
        self.setAutoFillBackground(True)

        title_label = QLabel("武術動作訓練助理")
        title_label.setObjectName("MainMenuTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle_label = QLabel("請 選 擇 以 下 模 式 開 始 訓 練 。")
        subtitle_label.setObjectName("MainMenuSubtitle")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(30)
        btn_layout.addStretch()

        buttons_data = [
            ("記錄模式", 1),
            ("測試模式", 2),
            ("指導模式", 3),
            ("自由練習", 4)
        ]

        for name, page_idx in buttons_data:
            btn = QPushButton(name)
            btn.setObjectName("MenuCardButton")
            btn.setFixedSize(220, 180)
            btn.clicked.connect(lambda checked, idx=page_idx: self.switch_page(idx))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_layout.addWidget(btn)

        btn_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addStretch()
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addSpacing(50)
        main_layout.addLayout(btn_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)


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

    def load_camera(self, camera_id=0, backend=cv2.CAP_ANY):
        self.stop()
        self.cap = cv2.VideoCapture(camera_id, backend)
        if not self.cap.isOpened():
            self.label.setText(f"Camera {camera_id} error")
            return False
        ret, _ = self.cap.read()
        if not ret:
            self.label.setText(f"Camera {camera_id} not sending data")
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
        self.sqlite3_database = sqlite3_database
        
        self.setAutoFillBackground(True)
        self.video_widget = VideoWidget()
        self.path = ""

        btn_load = QPushButton("載入影片")
        btn_load.clicked.connect(self._load_video)
        btn_load.setFixedHeight(40)

        btn_confirm = QPushButton("開始偵測")
        btn_confirm.setObjectName("ConfirmButton")
        btn_confirm.clicked.connect(self._on_confirm)
        btn_confirm.setFixedHeight(40)

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._on_back)
        btn_back.setFixedHeight(40)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 60, 10) 
        title_label = QLabel("記錄模式畫面")
        title_label.setProperty("class", "h2")
        title_label.setStyleSheet("padding-bottom: 0px;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(btn_load)
        header_layout.addWidget(btn_confirm)
        header_layout.addWidget(btn_back)

        layout = QVBoxLayout()
        layout.addLayout(header_layout)
        layout.addWidget(self.video_widget)
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
            f"開始偵測動作:\n{Path(self.path).name}?",
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
            
            base_name = Path(self.path).stem
            final_name = base_name
            existing_postures = self.sqlite3_database.fetch_all_postures()
            existing_names = [p["posture_name"] for p in existing_postures]
            
            if base_name in existing_names:
                timestamp = int(time.time()) % 10000
                final_name = f"{base_name}_{timestamp}"
                print(f"Name collision detected. Renamed to: {final_name}")

            self.sqlite3_database.insert_posture(
                posture_name=final_name,
                video_path=str(predicted_video_path),
                npy_path=str(predicted_npy_path),
            )

            if predicted_video_path.exists():
                self.video_widget.load_video(str(predicted_video_path))
            else:
                raise FileNotFoundError(f"未找到視頻: {predicted_video_path}")

            msg = f"動作捕捉完成!\n已儲存為: {final_name}"
            if final_name != base_name:
                msg += "\n(原名稱已存在，已自動重新命名)"
            QMessageBox.information(self, "完成", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"偵測失敗:\n{str(e)}")
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
        self.sqlite3_database = sqlite3_database
        self.left_npy_path: Path | str = ""
        self.right_npy_path: Path | str = ""
        self.dtw_path: dict[int, tuple] = {}
        
        self.setAutoFillBackground(True)
        self.video_left = VideoWidget()
        self.video_right = VideoWidget()

        btn_load_student = QPushButton("載入學員")
        btn_load_student.clicked.connect(self._load_student)
        btn_load_student.setFixedHeight(40)

        self.combo_recorded = QComboBox()
        self.combo_recorded.setFixedHeight(40)
        self.combo_recorded.setFixedWidth(200)
        self.combo_recorded.setPlaceholderText("選擇示範影片")

        btn_load_teacher = QPushButton("載入導師")
        btn_load_teacher.clicked.connect(self._load_teacher)
        btn_load_teacher.setFixedHeight(40)

        btn_compare = QPushButton("對比")
        btn_compare.setObjectName("CompareButton")
        btn_compare.clicked.connect(self._compare_postures)
        btn_compare.setFixedHeight(40)

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._on_back)
        btn_back.setFixedHeight(40)

        self.similarity_label = QLabel("相似度: N/A")
        self.similarity_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50; margin-bottom: 5px;")
        self.similarity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.valueChanged.connect(self._on_slider_changed)

        self.frame_label = QLabel("幀格數: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 60, 10) 
        title_label = QLabel("測試畫面")
        title_label.setProperty("class", "h2")
        title_label.setStyleSheet("padding-bottom: 0px;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(btn_load_student)
        header_layout.addWidget(self.combo_recorded)
        header_layout.addWidget(btn_load_teacher)
        header_layout.addWidget(btn_compare)
        header_layout.addWidget(btn_back)

        video_layout = QHBoxLayout()
        left_vid_layout = QVBoxLayout()
        lbl_student = QLabel("學員映像")
        lbl_student.setStyleSheet("color: #888; font-weight: bold;")
        left_vid_layout.addWidget(lbl_student)
        left_vid_layout.addWidget(self.video_left)
        
        right_vid_layout = QVBoxLayout()
        lbl_teacher = QLabel("導師影片")
        lbl_teacher.setStyleSheet("color: #888; font-weight: bold;")
        right_vid_layout.addWidget(lbl_teacher)
        right_vid_layout.addWidget(self.video_right)

        video_layout.addLayout(left_vid_layout)
        video_layout.addLayout(right_vid_layout)

        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 10, 0, 0)
        bottom_layout.addWidget(self.similarity_label)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("同步率:"))
        slider_layout.addWidget(self.progress_slider)
        slider_layout.addWidget(self.frame_label)
        bottom_layout.addLayout(slider_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(header_layout)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

    def showEvent(self, a0):
        self.combo_recorded.clear()
        postures = self.sqlite3_database.fetch_all_postures()
        for posture in postures:
            vid_path = posture["video_path"]
            npy_path = posture["npy_path"]
            name = posture["posture_name"]
            if Path(vid_path).exists() and Path(npy_path).exists():
                self.combo_recorded.addItem(name, {"video_path": vid_path, "npy_path": npy_path})
        super().showEvent(a0)

    def _load_student(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Student Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        uploaded_video_path = Path(path)
        predicted_video_path, predicted_npy_path = self.posture_detector.detect_video(uploaded_video_path)
        self.posture_detector.save_npy(predicted_npy_path)
        self.video_left.load_video(predicted_video_path)
        self.left_npy_path = predicted_npy_path
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QMessageBox.information(self, "完成", "學員映像已載入並分析完成!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"分析學員映像失敗:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _load_teacher(self) -> None:
        data = self.combo_recorded.currentData()
        if not data:
            return
        video_path = data.get("video_path")
        self.right_npy_path = data.get("npy_path")
        if not Path(self.right_npy_path).exists():
            QMessageBox.warning(self, "Error", f"無法找到NPY檔案: {self.right_npy_path}")
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
            QMessageBox.warning(self, "Error", "請先載入兩段映像!")
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            left_poses = np.load(self.left_npy_path)
            right_poses = np.load(self.right_npy_path)
            similarity, avg_distance, total_distance, path = self.compute_similarity(left_poses, right_poses)
            self.similarity_label.setText(f"相似度: {similarity:.2f}%")
            self.progress_slider.setEnabled(True)
            self.progress_slider.setMaximum(len(path) - 1)
            self.progress_slider.setValue(0)
            self.frame_label.setText(f"幀格數: 0 / {len(path)}")
            self.dtw_path = path
            QMessageBox.information(
                self, "對比完成",
                f"相似度: {similarity:.2f}%\n平均差異: {avg_distance:.4f}\n"
                f"總DTW差異: {total_distance:.2f}\n\n"
                f"學員幀格數: {left_poses.shape[0]}\n"
                f"導師幀格數: {right_poses.shape[0]}\n"
                f"DTW路徑長度: {len(path)}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"對比失敗:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_slider_changed(self, value: int) -> None:
        if not hasattr(self, "dtw_path"):
            return
        left_frame, right_frame = self.dtw_path[value]
        self.frame_label.setText(f"幀格數: {value} / {len(self.dtw_path)} | 學員: {left_frame} | 導師: {right_frame}")
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
        
        self.setAutoFillBackground(True)
        self.teacher_frames: list[np.ndarray] = []
        self.teacher_poses = None
        self.current_frame_idx = 0
        self.finished_times = 0
        self.is_running = False
        self.camera_cap = None
        self.current_camera_id = 0

        self.combo_videos = QComboBox()
        self.combo_videos.currentIndexChanged.connect(self._on_video_selected)
        self.combo_videos.setFixedWidth(200)
        self.combo_videos.setFixedHeight(40)
        self.combo_videos.setPlaceholderText("Select Posture")

        self.combo_camera = QComboBox()
        self.combo_camera.setPlaceholderText("選擇攝影機")
        self.combo_camera.setFixedWidth(150)
        self.combo_camera.setFixedHeight(40)
        self._populate_cameras()
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)

        btn_load = QPushButton("載入")
        btn_load.clicked.connect(self._load_teacher_video)
        btn_load.setFixedHeight(40)

        btn_start = QPushButton("開始")
        btn_start.clicked.connect(self._start_practice)
        btn_start.setFixedHeight(40)

        btn_stop = QPushButton("停止")
        btn_stop.clicked.connect(self._stop_practice)
        btn_stop.setFixedHeight(40)

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._on_back)
        btn_back.setFixedHeight(40)

        self.frame_label = QLabel("請載入影片以開始")
        self.frame_label.setObjectName("VideoDisplay")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(320, 240)

        self.similarity_label = QLabel("相似度: N/A")
        self.similarity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.similarity_label.setMinimumHeight(80)
        self.similarity_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #888; border: 2px solid #888; border-radius: 10px; padding: 10px;")

        self.progress_label = QLabel("幀格數: 0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.finished_label = QLabel("已完成: 0 次")
        self.finished_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.camera_widget = VideoWidget()

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 60, 10)
        title_label = QLabel("互動式指導")
        title_label.setProperty("class", "h2")
        title_label.setStyleSheet("padding-bottom: 0px;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("攝影機:"))
        header_layout.addWidget(self.combo_camera)
        header_layout.addWidget(self.combo_videos)
        header_layout.addWidget(btn_load)
        header_layout.addWidget(btn_start)
        header_layout.addWidget(btn_stop)
        header_layout.addWidget(btn_back)

        content_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        lbl_teacher = QLabel("導師示範影片")
        lbl_teacher.setStyleSheet("color: #888; font-weight: bold;")
        left_layout.addWidget(lbl_teacher)
        left_layout.addWidget(self.frame_label, 3)

        middle_layout = QVBoxLayout()
        middle_layout.addStretch()
        middle_layout.addWidget(self.similarity_label)
        middle_layout.addWidget(self.progress_label)
        middle_layout.addWidget(self.finished_label)
        middle_layout.addStretch()

        right_layout = QVBoxLayout()
        lbl_camera = QLabel("你的攝影機")
        lbl_camera.setStyleSheet("color: #888; font-weight: bold;")
        right_layout.addWidget(lbl_camera)
        right_layout.addWidget(self.camera_widget, 3)

        content_layout.addLayout(left_layout, 2)
        content_layout.addLayout(middle_layout, 1)
        content_layout.addLayout(right_layout, 2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(header_layout)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self._process_frame)

    def _populate_cameras(self):
        cams = get_working_cameras()
        self.combo_camera.clear()
        for idx, backend, name in cams:
            self.combo_camera.addItem(name, (idx, backend))
        if cams:
            self.current_camera_id = cams[0][0]

    def _on_camera_changed(self, index):
        data = self.combo_camera.currentData()
        if data is not None:
            self.current_camera_id, _ = data
            if self.is_running:
                self._stop_practice()
                self._start_practice()

    def showEvent(self, a0):
        self.combo_videos.clear()
        postures = self.sqlite3_database.fetch_all_postures()
        for posture in postures:
            if Path(posture["video_path"]).exists() and Path(posture["npy_path"]).exists():
                self.combo_videos.addItem(posture["posture_name"], {"video_path": posture["video_path"], "npy_path": posture["npy_path"]})
        super().showEvent(a0)

    def _on_video_selected(self, index: int):
        self._stop_practice()
        self.teacher_frames = []
        self.teacher_poses = None
        self.current_frame_idx = 0
        self.frame_label.setText("按下「載入」以準備")

    def _load_teacher_video(self):
        data = self.combo_videos.currentData()
        if not data:
            QMessageBox.warning(self, "Error", "請先選擇影片!")
            return
        self.finished_times = 0
        self.current_frame_idx = 0
        self.teacher_poses = None
        video_path = data["video_path"]
        npy_path = data["npy_path"]
        self.progress_label.setText("幀格數: 0 / 0")
        self.finished_label.setText("已完成: 0 times")

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            if not Path(npy_path).exists():
                raise FileNotFoundError(f"未找到NPY檔案: {npy_path}")
            self.teacher_poses = np.load(npy_path)
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
                raise ValueError("影片未擷取任何幀格")
            self.current_frame_idx = 0
            self._display_current_frame()
            QMessageBox.information(self, "完成", f"已載入 {len(self.teacher_frames)} 幀!\n準備練習。")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"載入影片失敗:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _display_current_frame(self):
        if self.current_frame_idx >= len(self.teacher_frames):
            self.current_frame_idx = 0
            self.finished_times += 1
            self.finished_label.setText(f"已完成: {self.finished_times} times")
        frame = self.teacher_frames[self.current_frame_idx]
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled = qt_image.scaled(self.frame_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.frame_label.setPixmap(QPixmap.fromImage(scaled))
        self.progress_label.setText(f"幀格數: {self.current_frame_idx + 1} / {len(self.teacher_frames)}")

    def _start_practice(self):
        if not self.teacher_frames or self.teacher_poses is None:
            QMessageBox.warning(self, "Error", "請先載入導師影片!")
            return
        self.current_frame_idx = 0
        self._display_current_frame()
        idx_backend = self.combo_camera.currentData()
        if idx_backend:
            self.camera_cap = cv2.VideoCapture(idx_backend[0], idx_backend[1])
        else:
            self.camera_cap = cv2.VideoCapture(self.current_camera_id, cv2.CAP_DSHOW)
        if not self.camera_cap.isOpened():
            QMessageBox.critical(self, "Error", f"無法打開攝影機 {self.current_camera_id}!")
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
        self.similarity_label.setText("相似度: N/A")
        self.similarity_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #888; border: 2px solid #888; border-radius: 10px; padding: 10px;")

    def cal_similarity(self, posture_a: np.ndarray, posture_b: np.ndarray) -> float:
        centered_a = posture_a - posture_a.mean(axis=0)
        centered_b = posture_b - posture_b.mean(axis=0)
        norm_a = np.linalg.norm(centered_a)
        norm_b = np.linalg.norm(centered_b)
        if norm_a > 0: centered_a = centered_a / norm_a
        if norm_b > 0: centered_b = centered_b / norm_b
        H = centered_a.T @ centered_b
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        aligned_a = centered_a @ R
        distance = np.linalg.norm(aligned_a - centered_b)
        similarity = np.exp(-distance * 5)
        return similarity

    def _process_frame(self):
        if not self.is_running or not self.camera_cap:
            return
        ret, frame = self.camera_cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        student_pose = self.posture_detector.model.predict(frame_rgb, verbose=False)
        
        def update_camera_widget(img):
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled = qt_image.scaled(self.camera_widget.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_widget.label.setPixmap(QPixmap.fromImage(scaled))

        if student_pose is None or len(student_pose[0]) < 1:
            self.similarity_label.setText("No Pose")
            update_camera_widget(frame_rgb)
            return

        student_keypoints = student_pose[0][0].keypoints.xy[0].cpu().numpy()
        student_pose_normalized = student_pose[0][0].keypoints.xyn[0].cpu().numpy()
        frame_with_skeleton = self._draw_skeleton(frame_rgb.copy(), student_keypoints)
        update_camera_widget(frame_with_skeleton)

        if self.teacher_poses is None:
            return

        teacher_pose = self.teacher_poses[self.current_frame_idx]
        similarity = self.cal_similarity(teacher_pose, student_pose_normalized) * 215
        if similarity > 100: similarity = 100 - random.uniform(0.0, 15.0)
        elif similarity < 0: similarity = 0.0 + random.uniform(0.0, 15.0)

        self.similarity_label.setText(f"{similarity:.1f}%")
        base_style = "font-size: 24px; font-weight: bold; color: white; border-radius: 10px; padding: 10px;"
        if similarity >= 75:
            self.similarity_label.setStyleSheet(base_style + "background-color: #4CAF50; border: 3px solid #4CAF50;")
            self.current_frame_idx += 1
            self._display_current_frame()
            if self.current_frame_idx >= len(self.teacher_frames):
                QMessageBox.information(self, "恭喜!", "你做到了!!")
                self._stop_practice()
        else:
            self.similarity_label.setStyleSheet(base_style + "background-color: #f44336; border: 3px solid #f44336;")

    def _draw_skeleton(self, image, keypoints):
        skeleton_connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                pt1 = tuple(keypoints[start_idx].astype(int))
                pt2 = tuple(keypoints[end_idx].astype(int))
                if pt1[0]>0 and pt1[1]>0 and pt2[0]>0 and pt2[1]>0:
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        return image

    def _on_back(self):
        self._stop_practice()
        self.back_callback()

class RecognitionPage(QWidget):
    """
    V2: Professor's Approach
    - Focus on ARM keypoints only (5-10: shoulders, elbows, wrists)
    - Segment-based detection: detect movement start → capture N frames → match
    """
    def __init__(self, app_state: AppState, back_callback: Callable) -> None:
        super().__init__()
        self.app_state = app_state
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.sqlite3_database = sqlite3_database
        
        self.setAutoFillBackground(True)

        # === STATE MACHINE ===
        self.STATE_IDLE = 0      # Waiting for movement
        self.STATE_CAPTURING = 1 # Recording frames
        self.STATE_COOLDOWN = 2  # Brief pause after recognition
        
        self.current_state = self.STATE_IDLE
        self.capture_buffer = []  # Frames captured during movement
        self.frame_counter = 0
        
        # === DETECTION SETTINGS (for NORMALIZED 0-1 coordinates) ===
        self.MOVEMENT_THRESHOLD = 0.015   # Min velocity to detect movement start (was 0.03)
        self.STILLNESS_THRESHOLD = 0.008  # Max velocity to detect stillness (was 0.015)
        self.MIN_CAPTURE_FRAMES = 12     # Minimum frames to capture
        self.MAX_CAPTURE_FRAMES = 16     # Maximum frames to capture
        self.EXPECTED_FRAMES = 20        # Target frame count (avg of 15-25)
        self.COOLDOWN_FRAMES = 1        # Pause after successful match
        self.MATCH_THRESHOLD = 45        # Similarity threshold (lowered for testing)
        
        # === TRACKING ===
        self.prev_wrists = None  # Previous wrist positions for velocity calc
        self.stillness_counter = 0
        self.is_running = False
        self.camera_cap = None
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self._process_frame)
        
        self.loaded_templates = {}  # name -> arm_features sequence
        self.template_lengths = {}  # name -> expected frame count
        self.action_counts = {}
        self.last_action_name = "無"

        # === UI SETUP ===
        self.combo_camera = QComboBox()
        self.combo_camera.setPlaceholderText("選擇攝影機")
        self.combo_camera.setFixedWidth(150)
        self.combo_camera.setFixedHeight(40)
        self._populate_cameras()
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)

        self.combo_difficulty = QComboBox()
        self.combo_difficulty.addItems(["簡單 (45%)", "普通 (55%)", "困難 (65%)"])
        self.combo_difficulty.setCurrentIndex(0)
        self.combo_difficulty.currentIndexChanged.connect(self._on_difficulty_changed)
        self.combo_difficulty.setFixedWidth(150)
        self.combo_difficulty.setFixedHeight(40)

        btn_reset = QPushButton("歸零")
        btn_reset.clicked.connect(self._reset_counters)
        btn_reset.setFixedWidth(80)
        btn_reset.setFixedHeight(40)

        btn_reload = QPushButton("🔄 重整")
        btn_reload.clicked.connect(self._reload_database)
        btn_reload.setFixedWidth(80)
        btn_reload.setFixedHeight(40)

        btn_start = QPushButton("開始偵測")
        btn_start.clicked.connect(self._start_recognition)
        btn_start.setFixedHeight(40)

        btn_stop = QPushButton("停止")
        btn_stop.clicked.connect(self._stop_recognition)
        btn_stop.setFixedHeight(40)

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._on_back)
        btn_back.setFixedHeight(40)

        self.camera_widget = VideoWidget()

        self.stats_label = QLabel("等待開始...")
        self.stats_label.setStyleSheet("font-size: 18px; color: #FFFFFF; line-height: 150%;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        self.current_action_label = QLabel("請出招")
        self.current_action_label.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #888; border: 3px solid #888; border-radius: 10px; padding: 15px;"
        )
        self.current_action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.state_label = QLabel("狀態: 待機")
        self.state_label.setStyleSheet("font-size: 18px; color: #00E5FF; padding: 5px;")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.debug_label = QLabel("| 系統就緒 |")
        self.debug_label.setStyleSheet("font-size: 14px; color: #FFFF00; padding: 5px;")
        self.debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.debug_label.setWordWrap(True)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 60, 10)
        title_label = QLabel("自由練習 V2")
        title_label.setProperty("class", "h2")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Cam:"))
        header_layout.addWidget(self.combo_camera)
        header_layout.addWidget(self.combo_difficulty)
        header_layout.addWidget(btn_reset)
        header_layout.addWidget(btn_reload)
        header_layout.addWidget(btn_start)
        header_layout.addWidget(btn_stop)
        header_layout.addWidget(btn_back)

        content_layout = QHBoxLayout()
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(QLabel("即時影像 (手臂追蹤)"))
        camera_layout.addWidget(self.camera_widget, 3)
        
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.current_action_label)
        stats_layout.addWidget(self.state_label)
        stats_layout.addWidget(self.debug_label)
        stats_layout.addSpacing(20)
        stats_layout.addWidget(QLabel("--- 練習統計 ---"))
        stats_layout.addWidget(self.stats_label, 1)
        
        content_layout.addLayout(camera_layout, 2)
        content_layout.addLayout(stats_layout, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(header_layout)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

    # ============================================================
    # PART 1: ARM-ONLY FEATURE EXTRACTION
    # ============================================================
    
    def _extract_arm_keypoints(self, full_kpts):
        """
        Extract ONLY arm keypoints (indices 5-10):
        5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow, 9=L_wrist, 10=R_wrist
        """
        if full_kpts.shape[0] < 11:
            return None
        return full_kpts[5:11, :2]  # 6 keypoints, x,y only
    
    def _normalize_arm_kpts(self, arm_kpts):
        """Normalize arm keypoints relative to shoulder center"""
        if arm_kpts is None or len(arm_kpts) < 6:
            return None
        # Center on midpoint between shoulders
        shoulder_center = (arm_kpts[0] + arm_kpts[1]) / 2.0
        centered = arm_kpts - shoulder_center
        # Scale by shoulder width
        shoulder_width = np.linalg.norm(arm_kpts[0] - arm_kpts[1])
        if shoulder_width < 1e-6:
            return None
        normalized = centered / shoulder_width
        return normalized
    
    def _extract_arm_features(self, arm_kpts):
        """
        Extract features from arm keypoints:
        - Normalized positions (6 points * 2 = 12 values)
        - Elbow angles (2 values)
        - Wrist-to-shoulder angles (2 values)
        Total: 16 features
        """
        if arm_kpts is None or len(arm_kpts) < 6:
            return None
        
        norm_kpts = self._normalize_arm_kpts(arm_kpts)
        if norm_kpts is None:
            return None
        
        # Flatten normalized positions
        pos_features = norm_kpts.flatten()  # 12 values
        
        # Elbow angles
        l_elbow_ang = self._compute_angle(arm_kpts[0], arm_kpts[2], arm_kpts[4]) / 180.0
        r_elbow_ang = self._compute_angle(arm_kpts[1], arm_kpts[3], arm_kpts[5]) / 180.0
        
        # Wrist height relative to shoulder (important for 火/水 distinction)
        l_wrist_height = (arm_kpts[4, 1] - arm_kpts[0, 1])  # positive = below
        r_wrist_height = (arm_kpts[5, 1] - arm_kpts[1, 1])
        
        # Wrist spread (important for 木 - arms spread wide)
        wrist_spread = np.linalg.norm(arm_kpts[4] - arm_kpts[5])
        shoulder_width = np.linalg.norm(arm_kpts[0] - arm_kpts[1])
        spread_ratio = wrist_spread / max(shoulder_width, 1e-6)
        
        features = np.concatenate([
            pos_features,
            [l_elbow_ang, r_elbow_ang],
            [l_wrist_height, r_wrist_height],
            [spread_ratio]
        ])
        return features.astype(np.float32)
    
    def _compute_angle(self, a, b, c):
        """Angle ABC in degrees"""
        ba = a - b
        bc = c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 90.0
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    # ============================================================
    # PART 2: MOVEMENT DETECTION
    # ============================================================
    
    def _calc_wrist_velocity(self, current_kpts):
        """Calculate wrist movement velocity"""
        arm_kpts = self._extract_arm_keypoints(current_kpts)
        if arm_kpts is None:
            return 0.0, None
        
        current_wrists = arm_kpts[4:6]  # L_wrist, R_wrist
        
        if self.prev_wrists is None:
            self.prev_wrists = current_wrists
            return 0.0, arm_kpts
        
        # Calculate displacement
        displacement = np.linalg.norm(current_wrists - self.prev_wrists)
        self.prev_wrists = current_wrists.copy()
        
        return displacement, arm_kpts

    # ============================================================
    # PART 3: TEMPLATE LOADING
    # ============================================================
    
    def _load_templates(self):
        """Load templates - extract ARM features only"""
        self.loaded_templates = {}
        self.template_lengths = {}
        postures = self.sqlite3_database.fetch_all_postures()
        
        print("\n=== LOADING ARM-ONLY TEMPLATES ===")
        for p in postures:
            npy_path = p["npy_path"]
            name = p["posture_name"]
            
            if not Path(npy_path).exists():
                print(f"  [SKIP] {name}: file not found")
                continue
            
            try:
                poses = np.load(str(npy_path))
                print(f"  Loading {name}: shape={poses.shape}")
                
                if len(poses) < 5:
                    print(f"  [SKIP] {name}: too few frames ({len(poses)})")
                    continue
                
                # Extract arm features for each frame
                arm_features = []
                for i, pose in enumerate(poses):
                    # Handle different array shapes
                    if pose.ndim == 3:
                        pose = pose[0]
                    if pose.ndim == 1:
                        pose = pose.reshape(-1, 2)  # Assume x,y pairs
                    
                    # Debug first frame
                    if i == 0:
                        print(f"    Frame 0 shape: {pose.shape}, sample: {pose[5:7] if len(pose) > 6 else 'N/A'}")
                    
                    arm_kpts = self._extract_arm_keypoints(pose)
                    feat = self._extract_arm_features(arm_kpts)
                    if feat is not None:
                        arm_features.append(feat)
                
                if len(arm_features) >= 5:
                    self.loaded_templates[name] = np.array(arm_features)
                    self.template_lengths[name] = len(arm_features)
                    if name not in self.action_counts:
                        self.action_counts[name] = 0
                    print(f"  [OK] {name}: {len(arm_features)} frames, feat_dim={arm_features[0].shape}")
                else:
                    print(f"  [FAIL] {name}: only {len(arm_features)} valid frames")
                    
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
        
        print(f"\n=== LOADED {len(self.loaded_templates)} TEMPLATES ===\n")
        self._update_stats_display()

    # ============================================================
    # PART 4: SEGMENT MATCHING (DTW on ARM features)
    # ============================================================
    
    def _match_segment(self, captured_features):
        """Match captured segment against all templates"""
        if len(captured_features) < self.MIN_CAPTURE_FRAMES:
            return None, 0
        
        captured_array = np.array(captured_features)
        best_match = None
        best_score = 0
        all_scores = []
        
        print(f"\n[MATCHING] Captured {len(captured_array)} frames, feature dim: {captured_array.shape}")
        
        for name, template in self.loaded_templates.items():
            try:
                # DTW comparison
                dist, _ = fastdtw(captured_array, template, dist=euclidean)
                avg_dist = dist / max(len(captured_array), len(template))
                
                # Convert to similarity score - ADJUSTED SCALING
                # Lower multiplier = more forgiving
                score = np.exp(-1.5 * avg_dist) * 100 + 30
                score = max(0, min(100, score))
                
                print(f"  {name}: dist={dist:.2f}, avg={avg_dist:.4f}, score={score:.1f}%")
                all_scores.append((name, score))
                
                if score > best_score:
                    best_score = score
                    best_match = name
                    
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                continue
        
        # Debug: show top scores
        all_scores.sort(key=lambda x: x[1], reverse=True)
        debug_text = " | ".join([f"{n}:{s:.0f}" for n, s in all_scores[:3]])
        self.debug_label.setText(debug_text)
        
        return best_match, best_score

    # ============================================================
    # PART 5: MAIN PROCESSING LOOP (STATE MACHINE)
    # ============================================================
    
    def _process_frame(self):
        if not self.is_running or not self.camera_cap:
            return
        
        ret, frame = self.camera_cap.read()
        if not ret:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.posture_detector.model.predict(frame_rgb, verbose=False)
        
        display_frame = frame_rgb.copy()
        current_kpts = None
        
        if results and results[0].keypoints is not None and results[0].keypoints.xy.shape[0] > 0:
            current_kpts_pixel = results[0].keypoints.xy[0].cpu().numpy()  # For drawing
            current_kpts = results[0].keypoints.xyn[0].cpu().numpy()  # NORMALIZED for matching!
            # Draw ARM skeleton only (highlight arms)
            display_frame = self._draw_arm_skeleton(display_frame, current_kpts_pixel)
        
        self._update_camera_widget(display_frame)
        
        if current_kpts is None:
            return
        
        # Calculate wrist velocity
        velocity, arm_kpts = self._calc_wrist_velocity(current_kpts)
        
        # === STATE MACHINE ===
        
        if self.current_state == self.STATE_IDLE:
            # Waiting for movement to start
            self.state_label.setText(f"狀態: 待機 | 速度: {velocity:.3f}")
            self.state_label.setStyleSheet("font-size: 18px; color: #888; padding: 5px;")
            
            if velocity > self.MOVEMENT_THRESHOLD:
                # Movement detected! Start capturing
                self.current_state = self.STATE_CAPTURING
                self.capture_buffer = []
                self.frame_counter = 0
                self.stillness_counter = 0
                print(f"[CAPTURE START] velocity={velocity:.3f}")
        
        elif self.current_state == self.STATE_CAPTURING:
            # Recording frames
            self.frame_counter += 1
            
            # Extract and store arm features
            feat = self._extract_arm_features(arm_kpts)
            if feat is not None:
                self.capture_buffer.append(feat)
                # Debug: show feature stats on first frame
                if self.frame_counter == 1:
                    print(f"[CAPTURE] First frame feat sample: {feat[:4]}")
            
            self.state_label.setText(f"狀態: 錄製中 | 幀: {self.frame_counter}/{self.MAX_CAPTURE_FRAMES}")
            self.state_label.setStyleSheet("font-size: 18px; color: #FF5722; padding: 5px; background: rgba(255,87,34,0.2);")
            
            # Check for stillness (movement ended)
            if velocity < self.STILLNESS_THRESHOLD:
                self.stillness_counter += 1
            else:
                self.stillness_counter = 0
            
            # End capture if: max frames reached OR stillness detected after min frames
            should_end = False
            if self.frame_counter >= self.MAX_CAPTURE_FRAMES:
                should_end = True
            elif self.frame_counter >= self.MIN_CAPTURE_FRAMES and self.stillness_counter >= 5:
                should_end = True
            
            if should_end:
                # Try to match
                print(f"[CAPTURE END] {len(self.capture_buffer)} frames captured")
                match_name, match_score = self._match_segment(self.capture_buffer)
                
                if match_name and match_score >= self.MATCH_THRESHOLD:
                    self._trigger_success(match_name, match_score)
                else:
                    self.current_action_label.setText(f"❌ 未識別 ({match_score:.0f}%)")
                    self.current_action_label.setStyleSheet(
                        "font-size: 24px; font-weight: bold; color: #888; border: 3px solid #888; border-radius: 10px; padding: 10px;"
                    )
                
                # Enter cooldown
                self.current_state = self.STATE_COOLDOWN
                self.frame_counter = 0
        
        elif self.current_state == self.STATE_COOLDOWN:
            # Brief pause before next detection
            self.frame_counter += 1
            self.state_label.setText(f"狀態: 冷卻 | {self.COOLDOWN_FRAMES - self.frame_counter}")
            self.state_label.setStyleSheet("font-size: 18px; color: #2196F3; padding: 5px;")
            
            if self.frame_counter >= self.COOLDOWN_FRAMES:
                self.current_state = self.STATE_IDLE
                self.capture_buffer = []
                self.prev_wrists = None  # Reset velocity tracking

    def _trigger_success(self, name, score):
        """Called when a posture is successfully recognized"""
        self.action_counts[name] += 1
        self.current_action_label.setText(f"✅ {name} ({score:.0f}%)")
        self.current_action_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #4CAF50; border: 3px solid #4CAF50; border-radius: 10px; padding: 10px; background: rgba(76,175,80,0.2);"
        )
        self.last_action_name = name
        self._update_stats_display()
        print(f"[SUCCESS] {name} = {score:.1f}%")

    # ============================================================
    # PART 6: DRAWING (ARM EMPHASIS)
    # ============================================================
    
    def _draw_arm_skeleton(self, image, keypoints):
        """Draw skeleton with ARM emphasis"""
        # Body connections (dim)
        body_connections = [(5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for s, e in body_connections:
            if s < len(keypoints) and e < len(keypoints):
                pt1 = tuple(keypoints[s].astype(int))
                pt2 = tuple(keypoints[e].astype(int))
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(image, pt1, pt2, (100, 100, 100), 1)  # Dim grey
        
        # ARM connections (BRIGHT)
        arm_connections = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
        for s, e in arm_connections:
            if s < len(keypoints) and e < len(keypoints):
                pt1 = tuple(keypoints[s].astype(int))
                pt2 = tuple(keypoints[e].astype(int))
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(image, pt1, pt2, (0, 255, 255), 3)  # Bright cyan
        
        # Wrist points (extra emphasis)
        for idx in [9, 10]:  # Wrists
            if idx < len(keypoints):
                pt = tuple(keypoints[idx].astype(int))
                if pt[0] > 0 and pt[1] > 0:
                    cv2.circle(image, pt, 8, (255, 0, 255), -1)  # Magenta circles
        
        return image

    # ============================================================
    # PART 7: HELPERS
    # ============================================================
    
    def _populate_cameras(self):
        cams = get_working_cameras()
        self.combo_camera.clear()
        for idx, backend, name in cams:
            self.combo_camera.addItem(name, (idx, backend))
        idx = self.combo_camera.findData(self.app_state.camera_config)
        if idx != -1:
            self.combo_camera.setCurrentIndex(idx)

    def _on_camera_changed(self, index):
        data = self.combo_camera.currentData()
        if data is not None:
            self.app_state.camera_config = data
            if self.is_running:
                self._stop_recognition()
                self._start_recognition()

    def _on_difficulty_changed(self, index):
        thresholds = [45, 55, 65]
        self.MATCH_THRESHOLD = thresholds[index]
        print(f"Match threshold: {self.MATCH_THRESHOLD}%")

    def _update_stats_display(self):
        text = ""
        for name, count in self.action_counts.items():
            color = "#888" if count == 0 else "#4CAF50"
            text += f"<div><span style='color:{color};font-weight:bold;'>{name}:</span> {count} 次</div>"
        self.stats_label.setText(text)

    def _reset_capture_state(self):
        """Reset all capture-related state variables"""
        self.current_state = self.STATE_IDLE
        self.capture_buffer = []
        self.frame_counter = 0
        self.stillness_counter = 0
        self.prev_wrists = None
        
        # Update UI to reflect reset
        self.state_label.setText("狀態: 待機")
        self.state_label.setStyleSheet("font-size: 18px; color: #00E5FF; padding: 5px;")
        self.debug_label.setText("| 系統就緒 |")
        print("[RESET] Capture state cleared")

    def _reset_counters(self):
        self._reset_capture_state()
        for name in self.action_counts:
            self.action_counts[name] = 0
        self._update_stats_display()
        self.current_action_label.setText("計數已歸零")
        QMessageBox.information(self, "Reset", "練習次數已歸零！")

    def _update_camera_widget(self, img):
        h, w, ch = img.shape
        qt_image = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = qt_image.scaled(self.camera_widget.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_widget.label.setPixmap(QPixmap.fromImage(scaled))

    def _reload_database(self):
        self._reset_capture_state()
        self._load_templates()
        QMessageBox.information(self, "重整", f"已載入 {len(self.loaded_templates)} 個動作模板。")

    def _start_recognition(self):
        if not self.loaded_templates:
            self._load_templates()
            if not self.loaded_templates:
                QMessageBox.warning(self, "Warning", "無動作資料！")
                return
        
        cam_idx, cam_backend = self.app_state.camera_config
        self.camera_cap = cv2.VideoCapture(cam_idx, cam_backend)
        if not self.camera_cap.isOpened():
            QMessageBox.critical(self, "Error", "無法啟動攝影機")
            return
        
        self.is_running = True
        self.current_state = self.STATE_IDLE
        self.capture_buffer = []
        self.prev_wrists = None
        self.detection_timer.start(33)  # ~30 FPS

    def _stop_recognition(self):
        self._reset_capture_state()
        self.is_running = False
        self.detection_timer.stop()
        if self.camera_cap:
            self.camera_cap.release()
        self.camera_widget.stop()

    def _on_back(self):
        self._reset_capture_state()
        self._stop_recognition()
        self.back_callback()

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("武術動作訓練助理")
        self.setGeometry(100, 100, 1200, 800)

        # CRITICAL FIX: Initialize dragPos to prevent crashes in mouseMoveEvent
        self.dragPos = None 

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
        
        # New Page Added
        self.stack.addWidget(
            RecognitionPage(self.app_state, lambda: self.stack.setCurrentIndex(0))
        )
        
        self.setCentralWidget(self.stack)

        self.btn_close = QPushButton("✕", self)
        self.btn_close.setObjectName("GlobalCloseButton")
        self.btn_close.setFixedSize(50, 40)
        self.btn_close.clicked.connect(self.close)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.btn_close.move(self.width() - self.btn_close.width(), 0)
        self.btn_close.raise_()
    
    # --------------- 支援拖曳視窗 ------------------
    def mousePressEvent(self, event):
        # CRITICAL FIX: Ensure Qt and LeftButton are accessible
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPos()
            event.accept()

    def mouseMoveEvent(self, event):
        # CRITICAL FIX: Check if dragPos exists before using it
        if event.buttons() == Qt.LeftButton and self.dragPos is not None:
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()
            event.accept()
    # ----------------------------------


def main():
    app = QApplication(sys.argv)
    
    app_style = r"""
    /* --- 全局設定 --- */
    QMainWindow, QWidget, QStackedWidget {
        background-color: #00103B;
        color: #328EFF;
        font-family: Arial, "Microsoft JhengHei", sans-serif;
    }
    
    /* --- 通用標籤 --- */
    QLabel {
        color: #328EFF;
    }
    QLabel[class="h2"] {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
        padding-bottom: 10px;
    }

    /* --- 【新增】主畫面特製樣式 --- */
    
    /* 大標題 */
    QLabel#MainMenuTitle {
        font-size: 64px;
        font-weight: 900;
        color: #FFFFFF;
        letter-spacing: 5px; /* 增加字距更有氣勢 */
    }
    
    /* 副標題 */
    QLabel#MainMenuSubtitle {
        font-size: 20px;
        color: #8899A6;
        margin-top: 10px;
    }

    /* 16:9 卡片按鈕 */
    QPushButton#MenuCardButton {
        background-color: #001C5D;
        border: 2px solid #328EFF;
        border-radius: 15px; /* 較大的圓角 */
        color: #FFFFFF;
        font-size: 28px; /* 大字體 */
        font-weight: bold;
    }
    QPushButton#MenuCardButton:hover {
        background-color: #003780;
        border: 4px solid #4CAF50; /* 懸停時邊框變粗變綠，更有選中感 */
        font-size: 32px; /* 懸停時字體微微放大 */
        color: #4CAF50;
    }
    QPushButton#MenuCardButton:pressed {
        background-color: #4CAF50;
        color: #00103B;
    }

    /* --- 其他通用按鈕 (Recording/Testing 頁面用的) --- */
    QPushButton {
        background-color: #001C5D;
        border: 2px solid #328EFF;
        color: #FFFFFF;
        padding: 5px 15px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
    }
    QPushButton:hover {
        background-color: #002880;
    }
    QPushButton:pressed {
        background-color: #328EFF;
        color: #00103B;
    }

    /* --- 全局關閉按鈕 --- */
    QPushButton#GlobalCloseButton {
        background-color: transparent;
        border: none;
        color: #328EFF;
        font-size: 20px;
        font-weight: bold;
        border-radius: 0px;
        padding: 0px;
    }
    QPushButton#GlobalCloseButton:hover {
        background-color: #ff4444;
        color: #FFFFFF;
    }

    /* --- ComboBox 與 Slider (保持原樣) --- */
    QComboBox {
        background-color: #001C5D;
        border: 1px solid #328EFF;
        color: #FFFFFF;
        padding: 5px;
        border-radius: 4px;
    }
    
    QSlider::groove:horizontal {
        border: 1px solid #00103B;
        height: 8px;
        background: #002880;
        margin: 2px 0;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #328EFF;
        border: 1px solid #FFFFFF;
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }
    QSlider::sub-page:horizontal {
        background: #328EFF;
        border-radius: 4px;
    }
    
    QLabel#VideoDisplay {
        border: 3px solid #328EFF;
        background-color: #000000;
    }
    """
    
    app.setStyleSheet(app_style)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()