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
    def __init__(self, app_state: AppState, back_callback: Callable) -> None:
        super().__init__()
        self.app_state = app_state
        self.back_callback = back_callback
        self.posture_detector = pose_model
        self.sqlite3_database = sqlite3_database
        
        self.setAutoFillBackground(True)

        # Logic Variables
        self.is_running = False
        self.camera_cap = None
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self._process_frame)
        
        # --- ROLLING BUFFER ---
        # 90 frames = approx 3 seconds history
        self.motion_buffer = deque(maxlen=90) 
        self.loaded_templates = {} 
        self.action_counts = {}
        
        # --- Settings ---
        self.passing_threshold = 50  # <--- LOWERED TO 50 FOR EASIER DETECTION
        self.cooldown = 0
        self.check_interval = 0
        
        # Stability Filters
        self.stability_counter = 0     
        self.candidate_cache = None    
        
        self.last_action_name = "無"

        # --- UI SETUP ---
        self.combo_camera = QComboBox()
        self.combo_camera.setPlaceholderText("選擇攝影機")
        self.combo_camera.setFixedWidth(150)
        self.combo_camera.setFixedHeight(40)
        self._populate_cameras()
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)

        self.combo_difficulty = QComboBox()
        self.combo_difficulty.addItems(["簡單 (Easy)", "普通 (Normal)", "困難 (Hard)"])
        self.combo_difficulty.setCurrentIndex(0) # Default to Easy
        self.combo_difficulty.currentIndexChanged.connect(self._on_difficulty_changed)
        self.combo_difficulty.setFixedWidth(150)
        self.combo_difficulty.setFixedHeight(40)

        btn_reset = QPushButton("歸零")
        btn_reset.clicked.connect(self._reset_counters)
        btn_reset.setFixedWidth(80)
        btn_reset.setFixedHeight(40)
        btn_reset.setStyleSheet("background-color: #FF9800; border: none; color: white; font-weight: bold;")

        btn_reload = QPushButton("🔄 重整")
        btn_reload.clicked.connect(self._reload_database)
        btn_reload.setFixedWidth(80)
        btn_reload.setFixedHeight(40)
        btn_reload.setStyleSheet("background-color: #607D8B; border: none;")

        btn_clean = QPushButton("🧹 清理")
        btn_clean.clicked.connect(self._clean_invalid_records)
        btn_clean.setFixedWidth(80)
        btn_clean.setFixedHeight(40)
        btn_clean.setStyleSheet("background-color: #795548; border: none;")

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

        self.history_label = QLabel("上一個動作: 無")
        self.history_label.setStyleSheet("font-size: 16px; color: #FFFF00; background-color: rgba(255, 255, 255, 0.1); padding: 5px;")
        self.history_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.debug_label = QLabel("| 系統就緒 |")
        self.debug_label.setStyleSheet("font-size: 14px; color: #00E5FF; background-color: rgba(0,0,0,0.3); padding: 5px;")
        self.debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.debug_label.setWordWrap(True)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 60, 10)
        
        title_label = QLabel("自由練習")
        title_label.setProperty("class", "h2")
        title_label.setStyleSheet("padding-bottom: 0px;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Cam:"))
        header_layout.addWidget(self.combo_camera)
        header_layout.addWidget(self.combo_difficulty)
        header_layout.addWidget(btn_reset)
        header_layout.addWidget(btn_reload)
        header_layout.addWidget(btn_clean)
        header_layout.addWidget(btn_start)
        header_layout.addWidget(btn_stop)
        header_layout.addWidget(btn_back)

        content_layout = QHBoxLayout()
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(QLabel("即時影像"))
        camera_layout.addWidget(self.camera_widget, 3)
        
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.current_action_label)
        stats_layout.addWidget(self.history_label) 
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

    # --- 1. CORE MATH: GEOMETRY ---
    def _compute_angle(self, a, b, c):
        """ Calculates angle ABC (in degrees) """
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _extract_angular_features(self, kpts):
        angle_l_elbow = self._compute_angle(kpts[5], kpts[7], kpts[9])
        angle_r_elbow = self._compute_angle(kpts[6], kpts[8], kpts[10])
        angle_l_shoulder = self._compute_angle(kpts[11], kpts[5], kpts[7])
        angle_r_shoulder = self._compute_angle(kpts[12], kpts[6], kpts[8])
        features = np.array([
            angle_l_elbow / 180.0,
            angle_r_elbow / 180.0,
            angle_l_shoulder / 180.0,
            angle_r_shoulder / 180.0
        ], dtype=np.float32)
        return features

    def _process_sequence_features(self, sequence_poses):
        feats = []
        for pose in sequence_poses:
            if pose.shape[0] == 1 and pose.shape[1] == 17:
                pose = pose[0]
            kpts = pose[:, :2]
            f = self._extract_angular_features(kpts)
            feats.append(f)
        return np.array(feats)

    def _trim_static_frames(self, sequence_features, threshold=0.02):
        if len(sequence_features) < 10: return sequence_features
        diffs = np.linalg.norm(sequence_features[1:] - sequence_features[:-1], axis=1)
        active_indices = np.where(diffs > threshold)[0]
        if len(active_indices) < 5: return sequence_features
        start_idx = max(0, active_indices[0] - 2)
        end_idx = min(len(sequence_features), active_indices[-1] + 3)
        return sequence_features[start_idx:end_idx]

    # --- 2. LOAD TEMPLATES ---
    def _load_templates(self):
        self.loaded_templates = {}
        postures = self.sqlite3_database.fetch_all_postures()
        print("\n=== LOADING TEMPLATES (CALIBRATED) ===")
        
        for p in postures:
            npy_path = p["npy_path"]
            name = p["posture_name"]
            path_obj = Path(npy_path)
            
            if path_obj.exists():
                try:
                    poses = np.load(str(path_obj))
                    if len(poses) < 5: continue
                    feat_seq = self._process_sequence_features(poses)
                    trimmed_seq = self._trim_static_frames(feat_seq)
                    if len(trimmed_seq) > 5:
                        self.loaded_templates[name] = trimmed_seq
                        if name not in self.action_counts:
                            self.action_counts[name] = 0
                        print(f"  [OK] {name}: {len(feat_seq)} -> {len(trimmed_seq)} frames")
                except Exception as e: 
                    print(f"  [FAIL] {name}: {e}")
                    pass
        self._update_stats_display()

    # --- 3. RECOGNITION LOGIC ---
    def _judge_action_rules(self, motion_buffer):
        """ 
        Advanced Trajectory Judge (Relaxed for Usability)
        """
        if len(motion_buffer) < 15: return {}, "Buffering..."

        seq_array = np.array(motion_buffer) 
        
        # --- 1. EXTRACT TRAJECTORY ---
        start_pose = seq_array[0]
        end_pose = seq_array[-1]
        
        # Y-coordinates (Negative = UP)
        start_wrist_y = (start_pose[9, 1] + start_pose[10, 1]) / 2.0
        end_wrist_y = (end_pose[9, 1] + end_pose[10, 1]) / 2.0
        
        start_shoulder_y = (start_pose[5, 1] + start_pose[6, 1]) / 2.0
        end_shoulder_y = (end_pose[5, 1] + end_pose[6, 1]) / 2.0
        
        start_span = np.linalg.norm(start_pose[9] - start_pose[10])
        end_span = np.linalg.norm(end_pose[9] - end_pose[10])
        
        torso_size = np.linalg.norm(end_pose[5] - end_pose[11]) 
        
        # --- 2. DEFINE STATES ---
        end_height_diff = end_wrist_y - end_shoulder_y
        is_end_high = end_height_diff < -0.15 * torso_size # Relaxed 0.2 -> 0.15
        is_end_low = end_height_diff > 0.2 * torso_size
        is_end_level = not is_end_high and not is_end_low
        
        # Displacement (Start -> End)
        delta_y = end_wrist_y - start_wrist_y
        is_rising = delta_y < -0.1 * torso_size # Relaxed
        is_sinking = delta_y > 0.1 * torso_size # Relaxed
        
        # --- 3. GEOMETRY CHECKS ---
        l_elbow_ang = self._compute_angle(end_pose[5], end_pose[7], end_pose[9])
        r_elbow_ang = self._compute_angle(end_pose[6], end_pose[8], end_pose[10])
        avg_arm_angle = (l_elbow_ang + r_elbow_ang) / 2.0

        modifiers = {}

        # --- 4. APPLY RULES ---

        # 🔥 Fire: High + Rising OR Straight Arms
        if is_end_high:
            if is_rising or avg_arm_angle > 95: # Relaxed 105 -> 95
                modifiers['Fire'] = 40
            else:
                modifiers['Fire'] = -10
        else:
            modifiers['Fire'] = -50

        # 💧 Water: Low + Sinking
        if is_end_low:
            if is_sinking: 
                modifiers['Water'] = 50
            elif start_wrist_y > end_shoulder_y: 
                # If started low, assume it's Earth/Rest, penalize Water
                modifiers['Water'] = -10 
            else:
                modifiers['Water'] = 10 
        else:
            modifiers['Water'] = -50

        # 🌲 Wood: Wide + Level
        shoulder_width = np.linalg.norm(end_pose[5] - end_pose[6])
        if end_span > 1.2 * shoulder_width: # Relaxed 1.3 -> 1.2
            if is_end_level: 
                modifiers['Wood'] = 50
            else: 
                modifiers['Wood'] = -20
        else:
            modifiers['Wood'] = -50 

        # ⛰️ Earth: Low + Narrow + Static
        if is_end_low and end_span < 1.0 * shoulder_width:
            if not is_sinking: 
                modifiers['Earth'] = 50
            else:
                modifiers['Earth'] = -10 
        else:
            modifiers['Earth'] = -50

        # 🏆 Metal: High + Narrower?
        if is_end_high:
            modifiers['Metal'] = 30
        else:
            modifiers['Metal'] = -50

        move_str = "Rise" if is_rising else ("Sink" if is_sinking else "Stat")
        debug_info = f"{move_str}|Ht:{'HI' if is_end_high else ('LO' if is_end_low else 'LV')}|Ang:{avg_arm_angle:.0f}"
        
        return modifiers, debug_info

    def _scan_buffer(self):
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        if len(self.motion_buffer) < 20: 
            self.debug_label.setText("| Buffering... |")
            return

        buffer_array = np.array(self.motion_buffer)
        live_features = self._process_sequence_features(buffer_array)
        judge_mods, hud_stats = self._judge_action_rules(self.motion_buffer)
        
        live_features_mirror = live_features.copy()
        live_features_mirror[:, [0, 1]] = live_features_mirror[:, [1, 0]]
        live_features_mirror[:, [2, 3]] = live_features_mirror[:, [3, 2]]

        scores = []
        b_len = len(live_features)
        
        for name, template_seq in self.loaded_templates.items():
            t_len = len(template_seq)
            if b_len < t_len * 0.7: continue
            
            slice_len = min(b_len, int(t_len * 1.2))
            window_slice = live_features[-slice_len:]
            window_slice_mirror = live_features_mirror[-slice_len:]
            
            try:
                dist_norm, _ = fastdtw(window_slice, template_seq, dist=euclidean)
                dist_mirr, _ = fastdtw(window_slice_mirror, template_seq, dist=euclidean)
                avg_dist = min(dist_norm, dist_mirr) / max(len(window_slice), t_len)
                
                raw_score = np.exp(-3.5 * avg_dist) * 100 
                
                final_score = raw_score
                for key, mod in judge_mods.items():
                    if key in name: final_score += mod
                
                final_score = max(0, min(100, final_score))
                
                # SHOW SCORES > 10 SO USER SEES PROGRESS
                if final_score > 10: scores.append((name, final_score))
            except: continue

        scores.sort(key=lambda x: x[1], reverse=True)
        
        top_name = None
        top_score = 0
        if scores:
            top_name, top_score = scores[0]

        # VISUAL DEBUGGING
        candidates_text = ""
        for name, score in scores[:2]:
            if score > self.passing_threshold:
                color = "#00FF00" # Green (Pass)
            else:
                color = "#888888" # Grey (Fail)
            candidates_text += f" <span style='color:{color};'>{name}:{score:.0f}</span>"
            
        status_color = "#AAA"
        if self.stability_counter > 0: status_color = "#FFFF00" # Yellow for locking
        
        self.debug_label.setText(f"<div style='font-size:12px; color:{status_color};'>{hud_stats}</div><div style='margin-top:4px;'>{candidates_text}</div>")

        # STABILITY LOGIC
        if top_score > self.passing_threshold:
            if top_name == self.candidate_cache:
                self.stability_counter += 1
            else:
                self.candidate_cache = top_name
                self.stability_counter = 1
                
            if self.stability_counter >= 2:
                self._trigger_success(top_name, top_score)
                self.stability_counter = 0 
                self.candidate_cache = None
        else:
            self.stability_counter = 0
            self.candidate_cache = None

    def _trigger_success(self, name, score):
        self.action_counts[name] += 1
        self.current_action_label.setText(f"✅ {name} ({score:.0f}%)")
        self.current_action_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50; border: 3px solid #4CAF50; border-radius: 10px; padding: 10px;")
        self.last_action_name = f"{name}"
        self.history_label.setText(f"上一個動作: {name} ({score:.0f}%)")
        self._update_stats_display()
        self.motion_buffer.clear() 
        self.cooldown = 15 
        self.debug_label.setText(f"| ✅ {name} HIT! |")

    # --- 4. PROCESS FRAME ---
    def _process_frame(self):
        if not self.is_running or not self.camera_cap: return
        ret, frame = self.camera_cap.read()
        if not ret: return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.posture_detector.model.predict(frame_rgb, verbose=False)
        display_frame = frame_rgb.copy()
        current_kpts = None
        if results and results[0].keypoints is not None and results[0].keypoints.xy.shape[0] > 0:
            current_kpts = results[0].keypoints.xy[0].cpu().numpy()
            display_frame = self._draw_skeleton(display_frame, current_kpts)
        self._update_camera_widget(display_frame)
        if current_kpts is None: return
        self.motion_buffer.append(current_kpts) 
        self.check_interval += 1
        if self.check_interval % 2 == 0: 
            self._scan_buffer()

    # --- HELPERS ---
    def _populate_cameras(self):
        cams = get_working_cameras()
        self.combo_camera.clear()
        for idx, backend, name in cams:
            self.combo_camera.addItem(name, (idx, backend))
        idx = self.combo_camera.findData(self.app_state.camera_config)
        if idx != -1: self.combo_camera.setCurrentIndex(idx)

    def _on_camera_changed(self, index):
        data = self.combo_camera.currentData()
        if data is not None:
            self.app_state.camera_config = data
            if self.is_running:
                self._stop_recognition()
                self._start_recognition()

    def _on_difficulty_changed(self, index):
        if index == 0: self.passing_threshold = 50
        elif index == 1: self.passing_threshold = 65
        elif index == 2: self.passing_threshold = 80
        print(f"Threshold: >{self.passing_threshold}%")

    def _update_stats_display(self):
        text = ""
        for name, count in self.action_counts.items():
            color = "#FFFFFF" if count == 0 else "#4CAF50"
            text += f"<div style='margin-bottom:5px;'><span style='color:{color}; font-weight:bold;'>{name}:</span> {count} 次</div>"
        self.stats_label.setText(text)

    def _reset_counters(self):
        for name in self.action_counts:
            self.action_counts[name] = 0
        self._update_stats_display()
        self.current_action_label.setText("計數已歸零")
        self.history_label.setText("上一個動作: 無")
        QMessageBox.information(self, "Reset", "練習次數已歸零！")

    def _update_camera_widget(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled = qt_image.scaled(self.camera_widget.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_widget.label.setPixmap(QPixmap.fromImage(scaled))

    def _draw_skeleton(self, image, keypoints):
        skeleton_connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                pt1 = tuple(keypoints[start_idx].astype(int))
                pt2 = tuple(keypoints[end_idx].astype(int))
                if pt1[0]>0 and pt1[1]>0 and pt2[0]>0 and pt2[1]>0:
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        return image

    def _reload_database(self):
        self._load_templates()
        QMessageBox.information(self, "重整完成", f"已重新載入 {len(self.loaded_templates)} 個動作模板。")

    def _clean_invalid_records(self):
        postures = self.sqlite3_database.fetch_all_postures()
        removed_count = 0
        for p in postures:
            if not Path(p["video_path"]).exists() or not Path(p["npy_path"]).exists():
                try: self.sqlite3_database.delete_posture(p["posture_name"]); removed_count += 1
                except: pass
        if removed_count > 0:
            self._reload_database()
            QMessageBox.information(self, "清理完成", f"已移除 {removed_count} 筆無效紀錄。")
        else:
            QMessageBox.information(self, "清理", "沒有發現無效紀錄。")

    def _start_recognition(self):
        if not self.loaded_templates:
            self._load_templates()
            if not self.loaded_templates:
                QMessageBox.warning(self, "Warning", "無動作資料！請先到記錄模式錄製動作。")
                return
        
        cam_idx, cam_backend = self.app_state.camera_config
        try:
            self.camera_cap = cv2.VideoCapture(cam_idx, cam_backend)
            if not self.camera_cap.isOpened(): 
                raise Exception("Camera failed to open")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"無法啟動攝影機: {e}")
            return
            
        self.is_running = True
        self.motion_buffer.clear() 
        self.detection_timer.start(30)

    def _stop_recognition(self):
        self.is_running = False
        self.detection_timer.stop()
        if self.camera_cap: self.camera_cap.release()
        self.camera_widget.stop()
        self.current_action_label.setText("已停止")

    def _on_back(self):
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