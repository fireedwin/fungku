# 影片訓練應用程式 (KungFu Video Training App)

一個基於 AI 姿態識別的功夫/武術訓練應用程式，使用 YOLO 模型進行姿態檢測，並透過動態時間規整 (DTW) 演算法比較學生與教師的動作相似度。

An AI-powered kung fu/martial arts training application using YOLO models for pose detection and Dynamic Time Warping (DTW) algorithm to compare student and teacher movement similarity.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.10+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v12-red.svg)

## ✨ 特色功能 (Features)

### 🎥 錄製模式 (Recording Mode)

- 載入影片並自動進行姿態偵測
- 使用 YOLO 模型提取關鍵點數據
- 將檢測結果儲存為視覺化影片和 NumPy 陣列
- 自動儲存至資料庫以供後續使用

### 📊 測試模式 (Testing Mode)

- 同時載入學生影片和教師示範影片
- 使用 DTW 演算法計算動作相似度
- 提供同步播放功能，可逐幀對比分析
- 顯示詳細的相似度分數和統計數據
- 視覺化呈現 DTW 對齊路徑

### 🎯 指導模式 (Guiding Mode)

- 即時攝影機追蹤和姿態分析
- 與教師示範影片進行即時比對
- 動態相似度回饋（綠色 ≥ 75%，紅色 < 75%）
- 自動進度追蹤，完成時顯示恭喜訊息
- 記錄完成次數

## 🛠️ 技術架構 (Tech Stack)

- **前端框架**: PyQt6 - 現代化的 GUI 介面
- **深度學習**: Ultralytics YOLO (YOLOv12) - 高精度姿態檢測
- **姿態檢測**: 自訓練的姿態和手部模型
- **相似度計算**: FastDTW + 歐幾里得距離
- **影像處理**: OpenCV, NumPy, Pillow
- **資料庫**: SQLite3 - 輕量級資料儲存
- **科學計算**: SciPy, NumPy

## 📋 系統需求 (Requirements)

- **Python**: >= 3.12
- **作業系統**: Windows, macOS, Linux
- **硬體需求**:
  - 攝影機（用於指導模式）
  - 建議配備 GPU 以加速 YOLO 推論（可選）
- **儲存空間**: 至少 2GB 可用空間（用於模型和結果檔案）

## 🚀 安裝步驟 (Installation)

### 1. 克隆專案 (Clone the repository)

```bash
git clone <repository-url>
cd KungFu
```

### 2. 安裝相依套件 (Install dependencies)

**使用 uv (推薦):**

```bash
uv sync
```

**或使用 pip:**

```bash
pip install -r requirements.txt
```

**主要相依套件:**

```
ultralytics>=8.3.227
pyqt6>=6.10.0
opencv-python>=4.12.0.88
numpy
fastdtw>=0.3.4
scipy>=1.16.3
mediapipe>=0.10.14
pillow>=12.0.0
```

### 3. 準備模型檔案 (Prepare model files)

建立 `model/` 目錄並放置訓練好的 YOLO 模型:

```bash
mkdir model
```

將以下模型放入 `model/` 目錄:

```
model/
├── pose_model.pt    # 姿態檢測模型 (必需)
└── hand_model.pt    # 手部檢測模型 (可選)
```

> **注意**: 您需要自行訓練 YOLO 姿態檢測模型，或使用預訓練的模型。

### 4. 執行應用程式 (Run the application)

```bash
uv run main.py
```

## 📁 專案結構 (Project Structure)

```
KungFu/
│
├── main.py                 # 主程式入口，包含所有 UI 介面
│   ├── MainWindow          # 主視窗
│   ├── MainPage            # 主選單頁面
│   ├── RecordingPage       # 錄製模式頁面
│   ├── TestingPage         # 測試模式頁面
│   └── GuidingPage         # 指導模式頁面
│
├── helper/                 # 輔助模組
│   ├── __init__.py
│   ├── model.py           # YOLO 模型載入與推論
│   │   └── ModelLoader    # 模型載入器類別
│   └── database.py        # SQLite 資料庫操作
│       └── Database       # 資料庫類別
│
├── model/                 # YOLO 模型檔案目錄
│   ├── pose_model.pt      # 姿態檢測模型
│   └── hand_model.pt      # 手部檢測模型
│
├── result/                # 檢測結果輸出目錄
│   └── predict*/          # 自動生成的預測結果
│       ├── *.avi          # 視覺化影片
│       └── *.npy          # 關鍵點數據
│
├── posture.sqlite3        # 姿態資料庫
├── pyproject.toml         # 專案配置檔案
├── log.log                # 應用程式日誌
└── README.md              # 專案說明文件
```

## 💻 使用說明 (Usage Guide)

### 📹 錄製模式 (Recording Mode)

1. 在主選單點擊「**錄製**」按鈕
2. 點擊「**載入影片**」選擇教師示範影片 (支援 mp4, avi, mov, mkv)
3. 預覽影片確認無誤
4. 點擊「**確認並偵測姿態**」開始 AI 分析
5. 等待處理完成（處理時間取決於影片長度）
6. 系統自動將檢測結果儲存至資料庫
7. 可查看處理後的影片（標註關鍵點）

**輸出檔案:**

- 視覺化影片: `result/predict*/[uuid].avi`
- 關鍵點數據: `result/predict*/[uuid].npy`

### 🔬 測試模式 (Testing Mode)

1. 在主選單點擊「**測試**」按鈕
2. 點擊「**載入學生影片**」上傳學生練習影片（左側）
   - 系統會自動進行姿態檢測
3. 從下拉選單選擇教師示範影片（右側）
4. 點擊「**載入教師示範**」
5. 點擊「**比較姿態**」開始分析
6. 查看相似度分數和統計資訊
7. 使用滑桿同步播放，逐幀比對動作差異

**分析指標:**

- **相似度百分比**: 0-100%，越高表示動作越相似
- **平均距離**: DTW 演算法計算的平均距離
- **總 DTW 距離**: 累積的時間規整距離
- **影格對齊**: 顯示學生與教師影格的對應關係

### 🥋 指導模式 (Guiding Mode)

1. 在主選單點擊「**指導**」按鈕
2. 從下拉選單選擇教師示範影片
3. 點擊「**載入教師影片**」
4. 確認教師示範影片已載入（左側顯示第一幀）
5. 點擊「**開始練習**」啟動攝影機
6. 系統會即時顯示相似度:
   - **綠色背景 (≥75%)**: 動作正確，自動進入下一幀
   - **紅色背景 (<75%)**: 需要調整動作
7. 跟隨教師動作，逐幀完成練習
8. 完成所有動作後顯示「**恭喜你完成了！**」訊息
9. 點擊「**停止**」結束練習

**即時回饋:**

- 相似度百分比實時更新
- 當前影格進度顯示
- 完成次數記錄

## 🔬 核心演算法 (Core Algorithms)

### 姿態正規化 (Keypoint Normalization)

將不同尺度和位置的姿態標準化，使其可以進行比較:

```python
def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    正規化關鍵點座標
    - 移除平移差異（中心對齊）
    - 移除縮放差異（正規化尺度）
    """
    kpts = np.array(kpts, dtype=np.float32)

    # 只使用 x, y 座標
    if kpts.ndim == 3:
        kpts = kpts[:, :2]
    elif kpts.shape[-1] == 3:
        kpts = kpts[:, :2]

    # 計算中心點
    center = np.mean(kpts, axis=0)

    # 計算縮放比例
    scale = np.linalg.norm(kpts - center)

    # 正規化: 移除平移和縮放
    return (kpts - center) / scale
```

### DTW 相似度計算 (DTW Similarity Computation)

使用動態時間規整演算法比較兩個姿態序列:

```python
def compute_similarity(seq_a, seq_b):
    """
    計算兩個姿態序列的相似度
    - 使用 FastDTW 進行時間對齊
    - 使用歐幾里得距離作為距離度量
    - 將距離轉換為 0-100% 的相似度分數
    """
    # 正規化每個姿態
    seq_a = [normalize_keypoints(pose).flatten() for pose in seq_a]
    seq_b = [normalize_keypoints(pose).flatten() for pose in seq_b]

    # 執行 DTW 對齊
    distance, path = fastdtw(seq_a, seq_b, dist=euclidean)

    # 計算平均距離
    avg_distance = distance / max(len(seq_a), len(seq_b))

    # 轉換為相似度百分比 (0-100%)
    similarity = np.exp(-5 * avg_distance) * 100
    similarity = max(0.0, min(100.0, similarity))

    return similarity, avg_distance, distance, path
```

**演算法特點:**

- **時間不變性**: DTW 允許序列以不同速度執行
- **尺度不變性**: 正規化消除身高和距離差異
- **平移不變性**: 中心對齊消除位置差異

## 📊 資料庫結構 (Database Schema)

### posture 表 (姿態資料表)

儲存所有錄製的姿態影片資訊:

| 欄位名稱     | 資料類型 | 說明                 | 約束                       |
| ------------ | -------- | -------------------- | -------------------------- |
| id           | INTEGER  | 主鍵                 | PRIMARY KEY, AUTOINCREMENT |
| posture_name | TEXT     | 姿態名稱             | NOT NULL                   |
| video_path   | TEXT     | 影片檔案路徑         | -                          |
| npy_path     | TEXT     | NumPy 關鍵點資料路徑 | -                          |

**範例資料:**

```sql
INSERT INTO posture (posture_name, video_path, npy_path)
VALUES ('金手', './result/predict/abc-123.avi', './result/predict/abc-123.npy');
```

### score 表 (分數資料表)

儲存練習分數和時間記錄:

| 欄位名稱   | 資料類型 | 說明               | 約束                       |
| ---------- | -------- | ------------------ | -------------------------- |
| id         | INTEGER  | 主鍵               | PRIMARY KEY, AUTOINCREMENT |
| time       | INTEGER  | 練習時間（秒）     | -                          |
| score      | INTEGER  | 相似度分數 (0-100) | -                          |
| video_path | TEXT     | 對應的影片路徑     | -                          |

**範例資料:**

```sql
INSERT INTO score (time, score, video_path)
VALUES (120, 85, './result/predict/student-456.avi');
```

## 🎯 依賴套件詳細說明 (Dependencies)

| 套件名稱          | 版本        | 用途                              |
| ----------------- | ----------- | --------------------------------- |
| **ultralytics**   | >=8.3.227   | YOLO 模型框架，用於姿態檢測       |
| **PyQt6**         | >=6.10.0    | GUI 框架，建立圖形使用者介面      |
| **opencv-python** | >=4.12.0.88 | 影像和影片處理                    |
| **numpy**         | latest      | 數值計算和陣列操作                |
| **fastdtw**       | >=0.3.4     | 快速動態時間規整演算法            |
| **scipy**         | >=1.16.3    | 科學計算（歐幾里得距離等）        |
| **mediapipe**     | >=0.10.14   | Google 的姿態檢測解決方案（可選） |
| **pillow**        | >=12.0.0    | 影像處理輔助                      |
| **matplotlib**    | >=3.10.7    | 資料視覺化（可選）                |
| **scikit-image**  | >=0.25.2    | 影像處理演算法（可選）            |

### 開發依賴 (Dev Dependencies)

```
uv sync
```

## ⚙️ 配置說明 (Configuration)

### 攝影機設定

預設使用攝影機 ID 1。若需更改，請修改以下位置:

**VideoWidget.load_camera()** (main.py:92)

```python
self.cap = cv2.VideoCapture(1)  # 修改數字: 0=內建, 1=外接, 2=第二個外接
```

**GuidingPage.\_start_practice()** (main.py:638)

```python
self.camera_cap = cv2.VideoCapture(1)  # 同上
```

### 相似度閾值調整

預設當相似度 ≥ 75% 時進入下一幀。可在 **main.py:702** 調整:

```python
if similarity >= 75:  # 修改此數值 (建議範圍: 60-90)
    self.current_frame_idx += 1
    self._display_current_frame()
```

**閾值建議:**

- **60-70%**: 寬鬆模式，適合初學者
- **75-80%**: 標準模式（預設）
- **85-90%**: 嚴格模式，適合進階練習

### 相似度計算參數

在 **main.py:385** 的相似度計算公式中:

```python
similarity = np.exp(-5 * avg_distance) * 100  # -5 是敏感度參數
```

**調整敏感度:**

- 較小的值（如 -3）: 較寬鬆，相似度分數較高
- 較大的值（如 -7）: 較嚴格，相似度分數較低

### 檢測更新頻率

指導模式的檢測間隔在 **main.py:644**:

```python
self.detection_timer.start(60)  # 單位: 毫秒 (60ms ≈ 16.7 FPS)
```

**建議設定:**

- **30ms**: 高頻率更新（33 FPS），較耗 CPU
- **60ms**: 平衡模式（16.7 FPS，預設）
- **100ms**: 省電模式（10 FPS）

## 🐛 已知問題 (Known Issues)

1. **手部檢測模型效果不佳**

   - 狀態: 已在程式碼中註解 (main.py:201-205)
   - 原因: 訓練資料不足或模型架構需要調整
   - 建議: 專注於姿態檢測，手部檢測暫時停用

2. **攝影機權限**

   - 問題: 某些系統需要手動授予攝影機權限
   - 解決方案:
     - Windows: 設定 > 隱私權 > 相機
     - macOS: 系統偏好設定 > 安全性與隱私權 > 相機
     - Linux: 確認使用者在 `video` 群組中

3. **YOLO 模型未提供**

   - 問題: 專案不包含預訓練模型
   - 解決方案: 需要自行訓練或獲取 YOLO 姿態檢測模型

4. **記憶體使用**

   - 問題: 長時間使用可能佔用較多記憶體
   - 建議: 定期重啟應用程式

5. **影片格式相容性**
   - 支援: mp4, avi, mov, mkv
   - 某些編解碼器可能不支援
   - 建議: 使用 H.264 編碼的 MP4 檔案

## 🚀 效能優化建議 (Performance Optimization)

### GPU 加速

YOLO 模型支援 GPU 加速。確保安裝 CUDA 版本的 PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 模型量化

使用較小的 YOLO 模型（如 YOLOn）以提升速度:

```python
# helper/model.py
pose_model = ModelLoader("./model/yolov12n-pose.pt")  # n=nano, s=small, m=medium
```

### 影片解析度

降低輸入影片解析度可提升處理速度:

```python
# 在 model.predict() 中加入 imgsz 參數
self.predict = self.model.predict(
    video_path,
    show_boxes=False,
    save=True,
    project="./result",
    imgsz=640  # 預設, 可降至 320 或 480
)
```

## 📝 開發者資訊 (Developer Info)

**作者 (Author)**: AkinoAlice@TyrantRey

**開發環境:**

- Python 3.12+
- PyQt6
- Ultralytics YOLOv12

**貢獻者 (Contributors):**

- 歡迎提交 Pull Request

## 🎓 模型訓練指南 (Model Training Guide)

### 準備訓練資料

1. **收集影片資料**

   - 錄製各種功夫動作的影片
   - 建議每個動作至少 100 個樣本
   - 確保不同角度、光線、背景

2. **標註關鍵點**

   - 使用 [Roboflow](https://roboflow.com/) 或 [CVAT](https://cvat.org/)
   - 標註身體關鍵點（17 個 COCO 關鍵點）
   - 匯出為 YOLO 格式

3. **訓練模型**

   ```python
   from ultralytics import YOLO

   # 載入預訓練模型
   model = YOLO('yolov12n-pose.pt')

   # 開始訓練
   results = model.train(
       data='kungfu_pose.yaml',
       epochs=100,
       imgsz=640,
       batch=16
   )

   # 儲存模型
   model.save('model/pose_model.pt')
   ```

4. **資料集配置 (kungfu_pose.yaml)**

   ```yaml
   path: /path/to/dataset
   train: images/train
   val: images/val

   # 關鍵點定義
   kpt_shape: [17, 3] # 17 個關鍵點, 每個 (x, y, visibility)

   # 類別
   nc: 1
   names: ["person"]
   ```
