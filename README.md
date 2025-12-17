# 影片訓練應用程式 (KungFu Video Training App)

一個基於 AI 姿態識別的功夫/武術訓練應用程式，使用 YOLO 模型進行姿態檢測，並透過動態時間規整 (DTW) 演算法比較學生與教師的動作相似度。

An AI-powered kung fu/martial arts training application using YOLO models for pose detection and Dynamic Time Warping (DTW) algorithm to compare student and teacher movement similarity.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.10+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v12-red.svg)

## 🆕 最新更新 (Recent Updates)

### v2.0 - 重大改進

**🔢 完成次數追蹤改進 (Rep Counting Improvements)**

指導模式現在採用**狀態機系統**來更準確地追蹤完成次數：

- **WAITING → IN_PROGRESS → COMPLETED → WAITING** 狀態循環
- 需要連續 3 幀達標才開始計數（防止誤觸發）
- 追蹤整個動作序列的進度百分比
- 容許最多 10 幀的短暫失誤（不會重置進度）
- 新增視覺化進度條顯示完成百分比

**🎯 姿態識別改進 (Posture Identification Improvements)**

自由練習模式採用多項技術改進來提高識別準確度：

- **擴展特徵維度 (4D → 12D)**：新增膝蓋角度、髖部角度、軀幹傾斜度、手臂展開比例、手部高度等特徵
- **移除硬編碼規則**：刪除可能導致問題的五行手修正器系統
- **動態窗口匹配**：嘗試多種窗口大小（模板長度的 50%-150%）以找到最佳匹配
- **分數平滑處理**：使用最近 5 次分數的滑動平均，而非單幀決策
- **信心度過濾**：過濾低信心度的關鍵點並從鄰近點進行插值
- **運動狀態機**：防止靜止時計數，並加入適當的冷卻期

---

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
- **狀態機追蹤系統**：精確記錄完成次數
- **視覺化進度條**：顯示當前動作完成百分比
- 自動進度追蹤，完成時顯示恭喜訊息

### 🆓 自由練習模式 (Free Practice Mode)

- **全自動動作識別**: 系統自動分析您的動作並與資料庫中的所有姿態進行比對，無需手動選擇目標。
- **12 維特徵向量**: 使用擴展的特徵集提高識別準確度，包含角度、比例和位置特徵。
- **智慧計數與評分**: 動作相似度達標後自動計數，並具備分數加成機制 (Gamification) 提升練習樂趣。
- **動態難度調整**: 提供「簡單、普通、困難」三種難度分級，適應不同階段的練習者。
- **誤判防禦機制**: 內建「移動門檻 (Movement Gate)」與「起始點檢查」，防止發呆或無效動作被誤計。
- **自動鏡像偵測**: 支援 Auto-Mirror 邏輯，無論面對鏡頭的方向為何，皆能準確識別。
- **分數平滑處理**: 使用滑動平均減少單幀誤判。

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
│   ├── GuidingPage         # 指導模式頁面
│   └── RecognitionPage     # 自由練習模式頁面
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
- **進度條**顯示當前動作完成百分比
- 完成次數記錄（使用狀態機精確追蹤）

**狀態機工作流程:**

```
WAITING (等待開始)
    ↓ 連續 3 幀達標
IN_PROGRESS (進行中)
    ↓ 完成所有幀
COMPLETED (完成)
    ↓ 自動重置
WAITING (等待下一次)
```

### 🤸 自由練習模式 (Free Practice Mode)

**核心功能:**

- **固定窗口識別 (Fixed-Window Recognition)**: 系統偵測到動作開始後，會自動鎖定並錄製固定長度（約 1-2 秒），確保完整捕捉「起手-發力-收招」的全過程。
- **12 維擴展特徵**: 包含肘部角度、膝蓋角度、髖部角度、軀幹傾斜度、手臂展開比例、手部高度等，有效區分相似動作。
- **動態窗口匹配**: 自動嘗試 50%-150% 的窗口大小，找到最佳匹配。
- **分數平滑**: 使用 5 幀滑動平均，減少誤判。

**智慧狀態機:**

- **IDLE**: 待機偵測微小晃動。
- **RECORDING**: 觸發後強制錄滿指定幀數。
- **EVALUATING**: 自動結算並顯示前三名相似動作。
- **COOLDOWN**: 防止重複計分。

**資料庫管理工具**: 內建「🔄 重整」與「🧹 清理」按鈕，可一鍵清除無效的檔案連結，解決檔案刪除後資料庫不同步的問題。

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

### 擴展特徵提取 (Extended Feature Extraction)

自由練習模式使用 12 維特徵向量進行更準確的識別：

```python
def extract_extended_features(kpts: np.ndarray) -> np.ndarray:
    """
    提取 12 維擴展特徵向量
    
    特徵包含:
    - 左右肘部角度 (2D)
    - 左右膝蓋角度 (2D)
    - 左右髖部角度 (2D)
    - 軀幹傾斜角度 (1D)
    - 手臂展開比例 (1D)
    - 左右手高度比例 (2D)
    - 腿部展開比例 (1D)
    - 整體姿態緊湊度 (1D)
    """
    features = []
    
    # 肘部角度
    left_elbow_angle = calculate_angle(kpts[5], kpts[7], kpts[9])
    right_elbow_angle = calculate_angle(kpts[6], kpts[8], kpts[10])
    features.extend([left_elbow_angle, right_elbow_angle])
    
    # 膝蓋角度
    left_knee_angle = calculate_angle(kpts[11], kpts[13], kpts[15])
    right_knee_angle = calculate_angle(kpts[12], kpts[14], kpts[16])
    features.extend([left_knee_angle, right_knee_angle])
    
    # ... 更多特徵
    
    return np.array(features, dtype=np.float32)
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

### 動態窗口匹配 (Dynamic Window Matching)

自動嘗試多種窗口大小以找到最佳匹配：

```python
def dynamic_window_match(live_buffer, template_seq):
    """
    使用動態窗口大小進行 DTW 匹配
    - 嘗試模板長度的 50% 到 150%
    - 返回最佳匹配結果
    """
    template_len = len(template_seq)
    best_distance = float('inf')
    best_scale = 1.0
    
    # 嘗試不同的窗口大小
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        window_size = int(template_len * scale)
        if window_size > len(live_buffer):
            continue
            
        # 取最近的 window_size 幀
        live_window = live_buffer[-window_size:]
        
        # 計算 DTW 距離
        distance, _ = fastdtw(live_window, template_seq, dist=euclidean)
        normalized_dist = distance / max(window_size, template_len)
        
        if normalized_dist < best_distance:
            best_distance = normalized_dist
            best_scale = scale
    
    return best_distance, best_scale
```

### 分數平滑處理 (Score Smoothing)

使用滑動平均減少單幀誤判：

```python
class ScoreSmoother:
    """分數平滑器，使用滑動平均"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.score_history = []
    
    def add_score(self, score):
        self.score_history.append(score)
        if len(self.score_history) > self.window_size:
            self.score_history.pop(0)
    
    def get_smoothed_score(self):
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)
```

### 智慧識別邏輯 (Smart Recognition Logic)

自由練習模式採用了進階的混合演算法以確保準確度：

1. **移動門檻 (Movement Gate)**: 計算時間窗口內的總位移量，若使用者處於靜止狀態，自動略過計算以節省資源並防止誤判。

```python
# 在 _recognize_action 裡面
avg_speed = self._calculate_motion(self.frame_buffer)

if avg_speed < 0.015:
   self.debug_label.setText(f"狀態: 靜止 (Motion: {avg_speed:.3f})")
   return  # <--- 這裡就是門檻，不動就直接擋掉
```

2. **自動鏡像匹配 (Auto-Mirroring)**: 系統同時計算原始骨架與水平翻轉骨架的 DTW 距離，取最佳值作為最終分數，解決了網路攝影機鏡像翻轉的問題。

```python
# 1. 創造翻轉版數據
for feat in live_seq_normal:
   feat_flipped = feat.copy()
   feat_flipped[0::2] = -feat_flipped[0::2] # 把 X 軸數值變負號 (翻轉)
   live_seq_flipped.append(feat_flipped)

# 2. 兩個都比對，取最小值 (min_dist)
dist_norm, _ = fastdtw(live_seq_normal, template_seq, dist=euclidean)
dist_flip, _ = fastdtw(live_seq_flipped, template_seq, dist=euclidean)
min_dist = min(dist_norm, dist_flip) # <--- 自動選比較像的那邊
```

3. **軀幹定錨正規化 (Torso-Anchored Normalization)**: 傳統的正規化會消除所有尺度差異，導致無法區分「手伸直」與「手彎曲」。我們改用 **脊椎長度 (肩膀中心到臀部中心)** 作為基準尺。

```python
# 核心邏輯
shoulder_center = (kpts[5] + kpts[6]) / 2.0
hip_center = (kpts[11] + kpts[12]) / 2.0
torso_size = np.linalg.norm(shoulder_center - hip_center)
normalized_pose = (kpts - hip_center) / torso_size
```

**優點**: 保留了手部前後伸縮的 **深度資訊 (Depth Cues)**，能準確識別「火手（推）」與「土手（抱）」的差異。

### 狀態機計數系統 (State Machine Counting System)

指導模式使用狀態機精確追蹤完成次數：

```python
class RepCounterStateMachine:
    """完成次數追蹤狀態機"""
    
    WAITING = "waiting"        # 等待開始
    IN_PROGRESS = "in_progress"  # 進行中
    COMPLETED = "completed"    # 完成
    
    def __init__(self):
        self.state = self.WAITING
        self.good_frame_count = 0
        self.bad_frame_tolerance = 10
        self.frames_completed = 0
        self.total_frames = 0
        self.rep_count = 0
    
    def update(self, is_good_frame, current_frame_idx, total_frames):
        self.total_frames = total_frames
        
        if self.state == self.WAITING:
            if is_good_frame:
                self.good_frame_count += 1
                if self.good_frame_count >= 3:  # 連續 3 幀達標
                    self.state = self.IN_PROGRESS
                    self.frames_completed = 1
            else:
                self.good_frame_count = 0
                
        elif self.state == self.IN_PROGRESS:
            if is_good_frame:
                self.frames_completed += 1
                if self.frames_completed >= total_frames:
                    self.state = self.COMPLETED
                    self.rep_count += 1
            else:
                self.bad_frame_tolerance -= 1
                if self.bad_frame_tolerance <= 0:
                    self.reset()
                    
        elif self.state == self.COMPLETED:
            self.reset()
    
    def reset(self):
        self.state = self.WAITING
        self.good_frame_count = 0
        self.bad_frame_tolerance = 10
        self.frames_completed = 0
    
    def get_progress(self):
        if self.total_frames == 0:
            return 0.0
        return self.frames_completed / self.total_frames * 100
```

**演算法特點:**

- **時間不變性**: DTW 允許序列以不同速度執行
- **尺度不變性**: 正規化消除身高和距離差異
- **平移不變性**: 中心對齊消除位置差異
- **狀態追蹤**: 精確記錄完成次數和進度

## 📊 資料庫結構 (Database Schema)

### posture 表 (姿態資料表)

儲存所有錄製的姿態影片資訊:

| 欄位名稱 | 資料類型 | 說明 | 約束 |
|----------|----------|------|------|
| id | INTEGER | 主鍵 | PRIMARY KEY, AUTOINCREMENT |
| posture_name | TEXT | 姿態名稱 | NOT NULL |
| video_path | TEXT | 影片檔案路徑 | - |
| npy_path | TEXT | NumPy 關鍵點資料路徑 | - |

**範例資料:**

```sql
INSERT INTO posture (posture_name, video_path, npy_path)
VALUES ('金手', './result/predict/abc-123.avi', './result/predict/abc-123.npy');
```

### score 表 (分數資料表)

儲存練習分數和時間記錄:

| 欄位名稱 | 資料類型 | 說明 | 約束 |
|----------|----------|------|------|
| id | INTEGER | 主鍵 | PRIMARY KEY, AUTOINCREMENT |
| time | INTEGER | 練習時間（秒） | - |
| score | INTEGER | 相似度分數 (0-100) | - |
| video_path | TEXT | 對應的影片路徑 | - |

**範例資料:**

```sql
INSERT INTO score (time, score, video_path)
VALUES (120, 85, './result/predict/student-456.avi');
```

## 🎯 依賴套件詳細說明 (Dependencies)

| 套件名稱 | 版本 | 用途 |
|----------|------|------|
| **ultralytics** | >=8.3.227 | YOLO 模型框架，用於姿態檢測 |
| **PyQt6** | >=6.10.0 | GUI 框架，建立圖形使用者介面 |
| **opencv-python** | >=4.12.0.88 | 影像和影片處理 |
| **numpy** | latest | 數值計算和陣列操作 |
| **fastdtw** | >=0.3.4 | 快速動態時間規整演算法 |
| **scipy** | >=1.16.3 | 科學計算（歐幾里得距離等） |
| **mediapipe** | >=0.10.14 | Google 的姿態檢測解決方案（可選） |
| **pillow** | >=12.0.0 | 影像處理輔助 |
| **matplotlib** | >=3.10.7 | 資料視覺化（可選） |
| **scikit-image** | >=0.25.2 | 影像處理演算法（可選） |

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

**GuidingPage._start_practice()** (main.py:638)

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

### 難度設定 (自由練習模式)

| 難度 | 通過閾值 | 穩定幀數 | 適用對象 |
|------|----------|----------|----------|
| 簡單 | 40 | 2 | 初學者 |
| 普通 | 50 | 3 | 一般練習 |
| 困難 | 60 | 4 | 進階練習 |

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

## 📜 更新日誌 (Changelog)

### v2.0.0
- 新增狀態機計數系統，精確追蹤完成次數
- 擴展特徵向量從 4D 到 12D
- 新增動態窗口匹配演算法
- 新增分數平滑處理
- 新增信心度過濾功能
- 移除硬編碼的五行手規則
- 新增視覺化進度條
- 改善整體識別準確度

### v1.0.0
- 初始版本
- 錄製、測試、指導、自由練習四種模式
- 基於 DTW 的姿態比較
- YOLO 姿態檢測
