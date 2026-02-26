# 🍎 M3 GPU 加速說明

## M3 晶片的 GPU

你的 MacBook Air M3 **確實有 GPU**！

### 硬體架構
```
M3 晶片 (統一記憶體架構)
├── CPU: 8核心
│   ├── 4個性能核心 (P-cores)
│   └── 4個效能核心 (E-cores)
├── GPU: 8-10核心 Apple Silicon GPU ⭐
│   └── 支援 Metal API
└── 統一記憶體: 16GB (CPU/GPU 共享)
```

### 與傳統 GPU 的差異

| 特性 | MacBook Air M3 | 獨立 GPU (NVIDIA/AMD) |
|------|----------------|---------------------|
| GPU 類型 | 整合式 | 獨立顯卡 |
| 記憶體 | 與 CPU 共享 16GB | 獨立 VRAM (6-24GB) |
| API | Metal | CUDA/ROCm |
| TensorFlow 支援 | tensorflow-metal | tensorflow-gpu |
| 功耗 | 低 (~15W) | 高 (~150-300W) |

---

## 安裝 GPU 加速

### 快速安裝
```bash
# 執行一鍵安裝腳本
./install_gpu_support.sh
```

### 手動安裝
```bash
# 1. 升級 pip
python3 -m pip install --upgrade pip

# 2. 安裝 TensorFlow for macOS
python3 -m pip install tensorflow-macos==2.15.0

# 3. 安裝 Metal GPU 插件 (關鍵！)
python3 -m pip install tensorflow-metal==1.1.0

# 4. 安裝其他依賴
python3 -m pip install numpy scipy soundfile pandas tqdm hydra-core
```

### 驗證安裝
```bash
python3 -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPU 裝置:', tf.config.list_physical_devices('GPU'))
"
```

**預期輸出：**
```
TensorFlow: 2.15.0
GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## GPU 加速效果

### 速度對比

| 任務 | CPU 模式 | GPU 模式 | 加速比 |
|------|----------|----------|--------|
| **矩陣運算** | 1.0x | 2-3x | 🚀🚀 |
| **卷積運算** | 1.0x | 3-5x | 🚀🚀🚀 |
| **模型訓練** | 1.0x | 2-4x | 🚀🚀 |

### Prototype 訓練時間
```
❌ 無 GPU 加速:  40-60 分鐘
✅ 有 GPU 加速:  15-25 分鐘
📊 加速比例:     ~2.5x
```

---

## 如何運作

### Metal Performance Shaders (MPS)
```
TensorFlow 操作
    ↓
tensorflow-metal 插件
    ↓
Metal API (Apple 的圖形框架)
    ↓
M3 GPU 核心 (8-10 核心)
    ↓
統一記憶體 (16GB)
```

### 記憶體管理
- **統一記憶體架構**: CPU 和 GPU 共享 16GB RAM
- **動態分配**: TensorFlow 自動管理 GPU 記憶體
- **無需手動設定**: Metal 會自動優化

---

## 常見問題

### Q1: 為什麼活動監視器看不到 GPU 使用率？
**A:** macOS 的活動監視器不顯示 Neural Engine/GPU 的深度學習負載。要監控可以用：
```bash
# 安裝 asitop (第三方工具)
pip install asitop
sudo asitop
```

### Q2: 第一次訓練很慢？
**A:** 正常！Metal 需要編譯 GPU 核心（JIT），第一次會慢，之後會加速：
```
第一個 epoch:  ~5 分鐘（編譯中）
後續 epoch:    ~2 分鐘（已優化）
```

### Q3: GPU 記憶體不足？
**A:** M3 與系統共享記憶體，如果 OOM：
```yaml
# 降低 batch_size
batch_size: 2  # 從 4 降到 2
```

### Q4: 如何確認正在使用 GPU？
**A:** 訓練時觀察：
```python
# 在訓練腳本開頭加入
import tensorflow as tf
print("可用 GPU:", tf.config.list_physical_devices('GPU'))

# 訓練時會看到
# "Created TensorFlow Lite XNNPACK delegate for CPU/Metal"
```

### Q5: 需要安裝 CUDA 嗎？
**A:** **不需要！** CUDA 是 NVIDIA GPU 專用，M3 用 Metal。

---

## 效能調優建議

### 針對 M3 優化的設定
```yaml
# conf_prototype.yaml (已為你配置)
batch_size: 4              # M3 最佳值
num_workers: 2             # 避免 I/O 瓶頸
mixed_precision: False     # M3 不支援，關閉
```

### 訓練時的最佳實踐
```bash
# 1. 關閉不必要的應用
# 2. 插上電源（必須）
# 3. 確保良好散熱
# 4. 使用 GPU 加速

# 監控效能
watch -n 5 'ps aux | grep python'
```

---

## 技術細節

### tensorflow-metal 做了什麼？

1. **攔截 TensorFlow 操作**
   ```
   tf.matmul(a, b)
   → Metal 檢測到矩陣乘法
   → 轉換為 Metal Shader
   → 在 GPU 上執行
   ```

2. **自動記憶體管理**
   - 在 CPU/GPU 間智能搬移數據
   - 最小化記憶體複製
   - 動態分配 GPU 記憶體

3. **核心融合優化**
   - 多個操作融合成一個 GPU 核心
   - 減少記憶體讀寫
   - 提升吞吐量

### 支援的操作
```python
✅ 卷積 (Conv2D)
✅ 全連接層 (Dense)
✅ 批次正規化 (BatchNorm)
✅ 啟動函數 (ReLU, ELU 等)
✅ LSTM/GRU
✅ 注意力機制
⚠️ 部分自定義操作可能退回 CPU
```

---

## 與其他 GPU 對比

### M3 vs NVIDIA RTX 3060

| 指標 | M3 (8核 GPU) | RTX 3060 | 勝者 |
|------|--------------|----------|------|
| 訓練速度 | 1.0x | 2-3x | 🏆 NVIDIA |
| 記憶體 | 16GB 統一 | 12GB VRAM | 🏆 M3 |
| 功耗 | 15W | 170W | 🏆 M3 |
| 價格 | 內建 | +$300 | 🏆 M3 |
| 便攜性 | 筆電 | 桌機 | 🏆 M3 |
| 生態系統 | Metal | CUDA | 🏆 NVIDIA |

**結論：M3 很適合學習和小規模訓練，不適合大規模生產！**

---

## 總結

### ✅ 你的 M3 可以做到：
- 運行深度學習訓練
- 使用 GPU 加速（透過 Metal）
- 訓練小到中型模型
- 快速原型驗證

### ❌ M3 不適合：
- 大規模生產訓練
- 超大模型（>1B 參數）
- 需要 CUDA 的專案
- 多 GPU 訓練

### 🎯 最佳使用場景：
```
✅ 學習和實驗
✅ 快速原型驗證（你現在要做的！）
✅ 小數據集訓練
✅ 推論部署
❌ 大規模訓練（考慮雲端 GPU）
```

---

## 立即開始

```bash
# 1. 安裝 GPU 支援
./install_gpu_support.sh

# 2. 驗證安裝
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. 開始訓練
./run_m3.sh
```

你的 M3 已經準備好了！🚀
