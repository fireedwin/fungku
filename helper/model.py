# Code by AkinoAlice@TyrantRey


from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
from ultralytics.engine.results import Results
from pathlib import Path
from uuid import uuid4
import numpy as np


# only processing frame to landmark
class ModelLoader:
    def __init__(self, model_path: str = "./model/pose_model.pt") -> None:
        self.model = YOLO(model_path)
        self.predict: list[Results] = []

    def detect_video(self, video_path: str | Path) -> tuple[Path, Path]:
        self.path = Path(video_path)
        self.predict = self.model.predict(
            video_path, show_boxes=False, save=True, project="./result"
        )

        self.uuid = str(uuid4())
        saved_predicted_path = sum(1 for _ in Path("./result").rglob("*") if _.is_dir())
        predicted_video_path = Path(
            f"./result/predict{str('' if saved_predicted_path == 1 else saved_predicted_path)}"
        ) / (self.uuid + ".avi")

        predicted_npy_path = Path(
            f"./result/predict{str('' if saved_predicted_path == 1 else saved_predicted_path)}"
        ) / (self.uuid + ".npy")

        yolo_output_path = Path(
            f"./result/predict{str('' if saved_predicted_path == 1 else saved_predicted_path)}"
        ) / (self.path.stem + ".avi")

        # rename to uuid format
        yolo_output_path.rename(predicted_video_path)

        return predicted_video_path, predicted_npy_path

    def save_npy(self, save_path: str | Path) -> np.ndarray:
        if self.predict is None:
            raise RuntimeError("Must call detect_video() before save_npy()")

        all_keypoints: list[np.ndarray] = []

        for result in self.predict:
            keypoints = result.keypoints
            if keypoints is None:
                continue

            if keypoints.shape[0] < 1:
                continue

            xyn = keypoints.xyn[0].cpu().numpy()
            all_keypoints.append(xyn)

        if len(all_keypoints) == 0:
            raise ValueError("No valid keypoints detected in video")

        keypoints_array = np.array(all_keypoints)
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(path)
        np.save(path, keypoints_array)

        return keypoints_array


hand_model = ModelLoader("./model/hand_model.pt")
pose_model = ModelLoader("./model/pose_model.pt")

if __name__ == "__main__":
    # hand_model = ModelLoader("./model/hand_model.pt")

    pose_model = ModelLoader("./model/pose_model.pt")

    predicted_video_path, predicted_npy_path = pose_model.detect_video(
        "./video/金手 - Trim.mp4"
    )
    pose_model.save_npy(predicted_npy_path)
