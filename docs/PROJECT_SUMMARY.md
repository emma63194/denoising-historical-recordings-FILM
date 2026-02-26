# 老電影降噪專案配置總結

## ✅ 已完成的修改

### 1. 配置文件修改

#### `conf/dset/dataset.yaml`
- ✅ 更新訓練/驗證/測試集路徑為三種資料集（AISHELL-3、FSD50K、MusicNet）
- ✅ 移除不需要的 `path_piano_test`、`path_strings_test`、`path_orchestra_test`、`path_opera_test`
- ✅ 更新噪音路徑為 Gramophone 黑膠唱片噪音資料集
- ✅ 更新測試資料路徑為老電影測試資料

#### `conf/conf.yaml`
- ✅ 更新 TensorBoard 日誌路徑為本地相對路徑 `experiments/tensorboard_logs`

### 2. 程式碼修改

#### `dataset_loader.py`
**新增功能：**
1. ✅ **智能噪音生成器** (`__noise_sample_generator`)
   - 自動檢測是否有 `info.csv`
   - 有 CSV：使用原始的結構化方法
   - 無 CSV：直接讀取目錄下所有 .wav 文件（適用於 Gramophone 資料集）

2. ✅ **隨機混合訓練資料** (`generator_train`)
   - 按比例隨機選擇資料集：60% AISHELL-3 / 20% FSD50K / 20% MusicNet
   - 使用 `np.random.choice` 實現概率採樣
   - 向後兼容：資料集不足 3 個時自動回退到原始行為

3. ✅ **更新所有函數的噪音生成器調用**
   - `generator_train`
   - `generate_val_data`
   - `generate_test_data`
   - `generate_paired_data_test_formal`

**錯誤修正：**
- ✅ 修正拼寫錯誤：`WRONG SAMPLE RATe` → `WRONG SAMPLE RATE`

### 3. 資料修復

#### `test_movie/audio_files.txt`
- ✅ 移除文件末尾多餘的 `%` 符號
- 確保格式正確，每行一個文件名

---

## 📋 使用前檢查清單

### 必須完成的事項：

1. **確認外接硬碟已連接**
   ```bash
   ls "/Volumes/SP SSD 120"
   ```

2. **檢查所有資料集路徑**
   ```bash
   python check_datasets.py
   ```
   應該看到：
   - AISHELL-3 訓練集：數千個文件
   - FSD50K 訓練集：數千個文件
   - MusicNet 訓練集：數百個文件
   - Gramophone 噪音：數千個文件
   - 測試電影：2 個文件

3. **確認 Python 環境**
   ```bash
   conda env list
   # 或
   pip list | grep tensorflow
   ```

---

## 🚀 訓練流程

### 1. 啟動訓練
```bash
python train.py
```

### 2. 監控訓練（可選）
```bash
tensorboard --logdir experiments/tensorboard_logs
```
然後打開瀏覽器訪問 `http://localhost:6006`

### 3. 推論/測試
```bash
python inference.py --audio <測試音頻路徑>
```

---

## 🔍 關鍵特性說明

### 1. 資料混合策略
訓練時，模型會看到：
- **60% 中文對白**（AISHELL-3）- 模擬老電影對話
- **20% 環境音效**（FSD50K）- 模擬環境聲、槍聲、腳步聲等
- **20% 古典音樂**（MusicNet）- 模擬背景音樂

### 2. 噪音類型
使用 Gramophone 黑膠唱片噪音，包含：
- 劈啪聲（pops and clicks）
- 表面噪音（surface noise）
- 刮擦聲（scratches）
- 78 轉唱片特有的噪音特性

### 3. 測試方式
- **合成測試**：乾淨音頻 + 人工添加噪音（SNR 可控）
- **真實測試**：實際老電影音軌（`test_movie/` 目錄）

---

## ⚠️ 注意事項

1. **外接硬碟速度**
   - 如果訓練速度慢，考慮將部分資料複製到內建硬碟
   - 或增加 `num_workers` 參數（目前為 10）

2. **記憶體使用**
   - 批次大小（`batch_size`）目前為 16
   - 如果記憶體不足，降低到 8 或 4

3. **訓練時長**
   - 配置為 73 個 epoch
   - 每個 epoch 1000 步
   - 預計訓練時間：視硬體而定

4. **資料集比例調整**
   - 如果想改變混合比例，修改 `dataset_loader.py` 第 418 行：
   ```python
   mixing_ratios = [0.6, 0.2, 0.2]  # 可改為 [0.5, 0.3, 0.2] 等
   ```

---

## 🛠 故障排除

### 問題：找不到資料集
**解決方法：**
1. 確認外接硬碟已掛載
2. 檢查路徑拼寫是否正確
3. 執行 `check_datasets.py` 查看詳細狀況

### 問題：訓練速度很慢
**解決方法：**
1. 減少 `batch_size`
2. 降低 `num_workers`
3. 考慮將資料複製到內建硬碟

### 問題：記憶體不足（OOM）
**解決方法：**
1. 降低 `batch_size` 從 16 → 8 → 4
2. 減少 `seg_len_s_train` 從 5 → 3 秒

### 問題：噪音生成器錯誤
**解決方法：**
- 確認 Gramophone 目錄下有 .wav 文件
- 檢查採樣率是否為 44100Hz

---

## 📊 預期結果

訓練完成後，模型應該能夠：
- ✅ 有效去除黑膠唱片噪音
- ✅ 保留對白清晰度
- ✅ 保持音樂完整性
- ✅ 處理環境音效而不產生失真

最終模型檔案會儲存在：
```
experiments/trained_model/checkpoint
```
