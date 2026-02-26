#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""檢查所有資料集路徑是否正確"""

import glob
import os

print('=== 檢查資料集路徑 ===\n')

# 1. AISHELL-3 語音資料集
print('1. AISHELL-3 (中文對白)')
aishell_train = '/Volumes/SP SSD 120/data_aishell3/train/wav/**/*.wav'
aishell_test = '/Volumes/SP SSD 120/data_aishell3/test/wav/**/*.wav'
aishell_train_files = glob.glob(aishell_train, recursive=True)
aishell_test_files = glob.glob(aishell_test, recursive=True)
print(f'   訓練集: {len(aishell_train_files)} 個文件')
print(f'   測試集: {len(aishell_test_files)} 個文件')

# 2. FSD50K 音效資料集
print('\n2. FSD50K (音效)')
fsd_train = '/Volumes/SP SSD 120/FSD50K/FSD50K.dev_audio/*.wav'
fsd_test = '/Volumes/SP SSD 120/FSD50K/FSD50K.eval_audio/*.wav'
fsd_train_files = glob.glob(fsd_train)
fsd_test_files = glob.glob(fsd_test)
print(f'   訓練集: {len(fsd_train_files)} 個文件')
print(f'   測試集: {len(fsd_test_files)} 個文件')

# 3. MusicNet 音樂資料集
print('\n3. MusicNet (古典音樂)')
music_train = '/Volumes/SP SSD 120/Musicnet/musicnet/train_data/*.wav'
music_test = '/Volumes/SP SSD 120/Musicnet/musicnet/test_data/*.wav'
music_train_files = glob.glob(music_train)
music_test_files = glob.glob(music_test)
print(f'   訓練集: {len(music_train_files)} 個文件')
print(f'   測試集: {len(music_test_files)} 個文件')

# 4. Gramophone 噪音資料集
print('\n4. Gramophone 黑膠噪音')
noise_dir = '/Users/shizukaryu/Documents/學校/論文/dataset/Gramophone_Record_Noise_Dataset'
if os.path.exists(noise_dir):
    noise_files = glob.glob(os.path.join(noise_dir, '**/*.wav'), recursive=True)
    print(f'   總共: {len(noise_files)} 個噪音文件')
else:
    print('   ❌ 目錄不存在')

# 5. 測試電影
print('\n5. 測試電影 (真實老電影)')
test_movie = '/Users/shizukaryu/Documents/學校/論文/dataset/test_movie'
if os.path.exists(test_movie):
    test_files = glob.glob(os.path.join(test_movie, '*.wav'))
    audio_list = os.path.join(test_movie, 'audio_files.txt')
    print(f'   總共: {len(test_files)} 個測試文件')
    print(f'   audio_files.txt 存在: {os.path.exists(audio_list)}')
else:
    print('   ❌ 目錄不存在')

print('\n=== 檢查完成 ===')
print('\n⚠️  警告說明：')
print('   - 如果某個資料集顯示 0 個文件，請確認：')
print('     1. 外接硬碟 "SP SSD 120" 是否已連接')
print('     2. 路徑是否正確')
print('     3. 該目錄下是否確實有 .wav 文件')
