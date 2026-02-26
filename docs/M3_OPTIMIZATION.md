# 🍎 MacBook Air M3 16GB 專用配置評估

## ✅ 硬體評估結果

### 你的硬體規格
```
處理器: Apple M3 (8核心 CPU)
GPU: 8-10核心 (Metal Performance Shaders)
記憶體: 16GB 統一記憶體
散熱: 無風扇設計（被動散熱）
儲存: 外接 SSD (SP SSD 120)
```

### 可行性分析
| 項目 | 評估 | 說明 |
|------|------|------|
| **CPU 性能** | ✅ 良好 | M3 性能核心足夠應付訓練 |
| **記憶體** | ✅ 充足 | 16GB 足夠，但需優化 batch size |
| **GPU 加速** | ⚠️ 需設定 | 需安裝 tensorflow-metal |
| **散熱** | ⚠️ 注意 | 無風扇會熱節流，需分段訓練 |
| **儲存 I/O** | ⚠️ 瓶頸 | 外接 SSD 可能限速 |

**結論：✅ 可行，但需要優化配置**

---

## 🔧 已優化的配置

我已經為你的 MacBook Air M3 優化了配置：

### 記憶體優化
```yaml
batch_size: 4          # 16→4 (安全的記憶體使用)
buffer_size: 100       # 1000→100 (減少記憶體壓力)
seg_len_s_train: 3     # 3秒片段 (減少記憶體佔用)
```

### 散熱優化（避免熱節流）
```yaml
epochs: 8              # 10→8 (更短訓練時間)
steps_per_epoch: 80    # 100→80 (單輪更快完成)
num_workers: 2         # 4→2 (減少 CPU 負載)
```

### 模型輕量化
```yaml
num_tfc: 1            # 2→1 (減少計算)
depth: 3              # 4→3 (更淺的網路)
use_SAM: False        # 關閉注意力機制 (減少參數)
```

**預計模型參數量：約減少 60-70%**

---

## ⚡ 性能預期

### 訓練時間估算
```
無 GPU 加速 (純 CPU):  約 40-60 分鐘
有 Metal GPU 加速:     約 15-25 分鐘
```

### 記憶體使用
```
系統: ~4GB
訓練: ~6-8GB (batch_size=4)
剩餘: ~4-6GB (安全緩衝)
```

### 熱管理
```
0-10 分鐘:  正常溫度，全速運行
10-20 分鐘: 開始升溫，可能輕微降頻
20+ 分鐘:   達到熱平衡，穩定降頻運行
```

---

## 🚀 推薦步驟

### 1. 安裝 Metal GPU 加速（強烈推薦）

```bash
# 安裝 TensorFlow 和 Metal 插件
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal==1.1.0

# 驗證安裝
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
"
```

**預期輸出：**
```
TensorFlow version: 2.15.0
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 2. 環境準備

```bash
# 確保外接 SSD 已掛載
ls "/Volumes/SP SSD 120"

# 檢查數據集
python3 check_datasets.py

# 關閉不必要的應用程式
# 釋放盡可能多的記憶體和 CPU
```

### 3. 啟動訓練

```bash
# 使用優化後的配置
./train_prototype.sh

# 或者直接執行
python3 train.py --config-name conf_prototype
```

### 4. 監控訓練（可選，但會佔用資源）

```bash
# 新開終端視窗
tensorboard --logdir experiments/tensorboard_logs

# 如果電腦太熱，建議先不開 TensorBoard
```

---

## 💡 最佳實踐

### 散熱管理
```bash
# 1. 訓練前
- 關閉瀏覽器、IDE 等耗資源應用
- 將 MacBook 放在散熱良好的平面
- 環境溫度保持涼爽（開冷氣）

# 2. 訓練中
- 不要蓋上螢幕（會降低散熱）
- 使用散熱架或風扇輔助
- 監控活動監視器看 CPU 使用率

# 3. 如果過熱
- 暫停訓練，讓機器冷卻 10-15 分鐘
- 減少 batch_size 到 2
- 或分成多次短訓練（每次 3-4 epochs）
```

### 記憶體管理
```bash
# 監控記憶體使用
watch -n 5 'vm_stat | grep "Pages free\|Pages active"'

# 如果記憶體不足 (swap 激增)
# 修改配置：
batch_size: 2          # 4→2
num_workers: 1         # 2→1
```

### I/O 優化
```bash
# 如果外接硬碟太慢
# 選項 1: 複製部分數據到內建硬碟
mkdir -p ~/temp_dataset
cp -r "/Volumes/SP SSD 120/data_aishell3/train/wav/SSB0005" ~/temp_dataset/

# 然後修改 dataset_prototype.yaml:
# path_music_train: ["~/temp_dataset/SSB0005/*.wav"]

# 選項 2: 使用 Thunderbolt 接口（如果有）
```

---

## 📊 效能基準測試

### 先跑一個最小測試
```bash
# 創建最小測試配置 (1 epoch, 10 steps)
cat > test_hardware.py << 'EOF'
import tensorflow as tf
import time
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# 簡單測試
start = time.time()
x = tf.random.normal((4, 256, 1025, 2))
for _ in range(10):
    y = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    y = tf.keras.layers.ReLU()(y)
print(f"10 iterations took: {time.time() - start:.2f}s")
print("✅ GPU 測試完成" if time.time() - start < 5 else "⚠️ 可能沒有使用 GPU")
EOF

python3 test_hardware.py
```

---

## ⚠️ 注意事項

### 已知限制
1. **訓練速度** - 比有獨立 GPU 的桌機慢 5-10 倍
2. **熱節流** - 長時間訓練會降頻到 ~70% 性能
3. **記憶體** - 無法同時開啟大量應用程式
4. **電池** - 必須插電使用

### 不適合的場景
❌ 長時間訓練（超過 2 小時）  
❌ 大模型訓練（需要 >16GB RAM）  
❌ 批量實驗（需要反覆訓練）

### 適合的場景
✅ 快速原型驗證（10-30 分鐘）  
✅ 小規模測試  
✅ 概念驗證  
✅ 學習和實驗

---

## 🎯 分階段策略（推薦）

### 階段 1: 超快速測試（5 分鐘）
```bash
# 修改 conf_prototype.yaml:
epochs: 2
steps_per_epoch: 20
batch_size: 4

# 目標：確認流程能跑通
```

### 階段 2: 快速原型（15-25 分鐘）
```bash
# 使用當前優化配置
epochs: 8
steps_per_epoch: 80
batch_size: 4

# 目標：驗證基本降噪效果
```

### 階段 3: 完整訓練（考慮租用雲端 GPU）
```
Google Colab Pro / Kaggle / Paperspace
- 更強 GPU (T4/V100)
- 更快訓練速度
- 無散熱問題
- 費用：約 $10-20/月
```

---

## 📋 MacBook Air M3 檢查清單

開始前：
- [ ] 已安裝 tensorflow-macos + tensorflow-metal
- [ ] 外接 SSD 已連接且可讀取
- [ ] 關閉其他耗資源應用
- [ ] 插上電源（必須）
- [ ] 環境涼爽（重要）
- [ ] 空閒記憶體 > 8GB

訓練中：
- [ ] 監控活動監視器（CPU、記憶體）
- [ ] 注意機身溫度（太燙要休息）
- [ ] 觀察 swap 使用（不應該激增）
- [ ] Loss 正常下降
- [ ] 沒有錯誤訊息

---

## 🆘 故障排除

### 問題：記憶體不足
```bash
# 解決方法
batch_size: 2   # 改為 2
num_workers: 1  # 改為 1
seg_len_s_train: 2  # 改為 2 秒
```

### 問題：太熱/風扇狂轉（等等，M3 Air 沒風扇）
```bash
# 機身太燙時
1. 暫停訓練 (Ctrl+C)
2. 等 10-15 分鐘冷卻
3. 降低負載重新開始

# 修改配置
batch_size: 2
num_workers: 1
```

### 問題：訓練太慢
```bash
# 確認 GPU 加速
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 如果沒有 GPU，安裝 Metal
pip install tensorflow-metal
```

### 問題：外接硬碟太慢
```bash
# 複製數據到內建硬碟（如果空間足夠）
# 或考慮使用更少的訓練數據
```

---

## 📈 預期結果

### 成功指標
- ✅ 完成 8 個 epoch (~15-25 分鐘)
- ✅ Loss 從 0.08 降到 0.02-0.03
- ✅ 記憶體穩定在 8-10GB
- ✅ 機身溫暖但不燙手
- ✅ 推論能產生降噪音頻

### 效果評估
由於是輕量模型+少量數據：
- 降噪效果：★★★☆☆ (可見效果，但不完美)
- 語音清晰度：★★★★☆ (基本保留)
- 訓練速度：★★★☆☆ (可接受)

**這是驗證流程，不是最終模型！**

---

## 結論

**MacBook Air M3 16GB 完全可行！** 🎉

但建議：
1. **先跑 prototype 驗證** (15-25 分鐘)
2. **確認效果和流程**
3. **如需更好效果，考慮雲端 GPU 訓練正式模型**

你的配置已經優化完成，可以直接開始！
