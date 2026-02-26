# Historical Audio Denoising for Film Restoration

A deep learning-based audio denoising system adapted from the [Two-stage U-Net architecture](https://github.com/eloimoliner/denoising-historical-recordings) for restoring old movie soundtracks.

## Overview

This project adapts a gramophone record denoising model for film audio restoration. The system uses a multi-stage U-Net neural network trained on diverse audio types (speech, sound effects, music) to remove noise while preserving signal quality.

## Key Features

- **Multi-domain Training**: Combines speech (60%), sound effects (20%), and music (20%) datasets to handle diverse film soundtracks
- **Adjustable Denoising Strength**: Control denoising intensity from 0.0 to 1.0 without retraining
- **Optimized SNR Range**: Trained on 10-25 dB SNR, matching typical old film audio conditions
- **Automatic File Naming**: Output files include strength suffix to prevent overwrites

## Training History

Four complete training runs were conducted to optimize the model for film audio restoration:

### Training Run 1: Initial Prototype
**Date**: 2026-02-03  
**Configuration**: 8 epochs, 80 steps/epoch, SNR 2-20 dB  
**Results**:
- Validation MAE: 0.117 (Epoch 8)
- Training time: ~76 minutes
- Status: ❌ Model not saved (checkpoint configuration issue)

**Key Findings**:
- Fast convergence: Val MAE improved from 0.29 → 0.117 (60% improvement)
- Stable training without overfitting
- Established baseline performance

---

### Training Run 2: Checkpoint Fix
**Date**: 2026-02-03  
**Configuration**: 8 epochs, 80 steps/epoch, SNR 2-20 dB  
**Results**:
- Validation MAE: 0.129
- Training time: ~80 minutes
- Status: ✅ Model saved successfully

**Key Findings**:
- ❌ **Critical Issue**: Over-denoising problem discovered
- Model removed ~90% of original signal
- Cause: SNR 2-5 dB training samples too noisy, model learned to output near-zero as safest strategy
- Test correlation (original vs residual): 0.9982 (residual contained most of the signal)

**Analysis**: The extremely low SNR range (2-20 dB) caused the model to be overly aggressive in noise removal, treating the signal itself as noise.

---

### Training Run 3: SNR Adjustment
**Date**: 2026-02-04  
**Configuration**: 8 epochs, 80 steps/epoch, **SNR 10-25 dB** (adjusted)  
**Results**:

| Epoch | Train MAE | Val MAE | Note |
|-------|-----------|---------|------|
| 1 | 0.2774 | 0.2962 | Initial |
| 2 | 0.1686 | 0.2535 | Fast convergence |
| 3 | 0.1539 | 0.2112 | - |
| 4 | 0.1274 | 0.1676 | - |
| 5 | 0.1430 | **0.1360** | **Best model** ✅ |
| 6 | 0.0642 | 0.1442 | Val loss increased |
| 7 | 0.0481 | 0.1308 | - |
| 8 | 0.0479 | 0.1293 | Final epoch |

**Training time**: ~69 minutes  
**Status**: ✅ Model saved

**Key Findings**:
- Val MAE improved 56% (0.296 → 0.129)
- SNR adjustment from 2-20 to 10-25 dB more realistic for film audio
- ⚠️ **User Feedback**: No significant improvement in actual audio quality
- **Problem**: Similar perceptual quality to Run 2 despite different SNR range

**Analysis**: 
- Noise type mismatch: Gramophone noise ≠ Film audio noise
- MAE loss does not capture perceptual quality
- Model architecture may need longer context windows

---

### Training Run 4: Increased Scale
**Date**: 2026-02-24 to 2026-02-25  
**Configuration**: **20 epochs, 200 steps/epoch**, SNR 10-25 dB  
**Validation Set Expansion**: 150 files → **628 files** (4.2× increase)

**Dataset Changes**:
- Speech: AISHELL-3 6 validation speakers (full set)
- Sound effects: FSD50K eval (500 files, limited from 9,692)
- Music: MusicNet test (full set)

**Results**:

| Epoch | Train MAE | Val MAE | Note |
|-------|-----------|---------|------|
| 1-5 | Decreasing | Decreasing | Fast initial convergence |
| 10 | ~0.10 | ~0.15 | Mid-training plateau |
| 15 | ~0.09 | ~0.155 | Stable validation loss |
| 20 | ~0.08 | **0.156** | **Final model** ✅ |

**Training time**: ~8.5 hours  
**Status**: ✅ Model saved

**Key Findings**:
- ✅ Stable training: No overfitting over 20 epochs
- ✅ Volume preservation: Fixed the signal removal issue from Runs 2-3
- ❌ **User Feedback**: "Limited denoising effect, almost like no denoising at all"
- **MAE Paradox**:
  - Runs 2-3: Lower Val MAE (0.129) but over-denoising
  - Run 4: Higher Val MAE (0.156) but under-denoising
  - **Conclusion**: MAE loss does not reflect perceptual quality

**Analysis**:
1. **Data diversity backfired**: 4.2× more diverse validation data made model conservative
2. **Root causes persist**:
   - Gramophone noise ≠ Film audio noise (fundamental mismatch)
   - MAE optimization ≠ Perceptual quality
   - Lack of real paired film audio data (noisy → clean)
3. **Model behavior**: Chose "do less, make fewer errors" strategy to minimize MAE across all samples

---

## Training Insights Summary

| Run | SNR (dB) | Val MAE | Training Scale | Outcome |
|-----|----------|---------|----------------|---------|
| 1 | 2-20 | 0.117 | 8 ep × 80 steps | Model not saved |
| 2 | 2-20 | 0.129 | 8 ep × 80 steps | Over-denoising (90% signal removal) |
| 3 | 10-25 | 0.129 | 8 ep × 80 steps | No improvement in audio quality |
| 4 | 10-25 | 0.156 | 20 ep × 200 steps | Under-denoising (minimal effect) |

**Key Learnings**:
- SNR range adjustment (2-20 → 10-25 dB) did not improve perceptual quality
- Increasing training scale led to more conservative model behavior
- **Critical finding**: MAE loss is not suitable for measuring denoising quality
- Noise type mismatch is the fundamental problem

Detailed analysis: [docs/TRAINING_SUMMARY.md](docs/TRAINING_SUMMARY.md)

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- CUDA 10.1+ (for NVIDIA GPUs, optional)

### Setup

**Option 1: pip (recommended)**
```bash
pip install tensorflow==2.13.0
pip install -r requirements.txt
```

**Option 2: conda**
```bash
conda env update -f environment.yml
conda activate historical_denoiser
```

## Usage

### Inference

Denoise audio files using pre-trained model:

```bash
# Basic usage
python inference.py --config-name conf_prototype \
    inference.audio=your_audio.wav

# Adjust denoising strength (0.0 = no denoising, 1.0 = full denoising)
python inference.py --config-name conf_prototype \
    inference.audio=your_audio.wav \
    inference.denoising_strength=0.5
```

**Output files:**
- `your_audio_denoised_s5.wav` - Denoised audio
- `your_audio_residual_s5.wav` - Removed noise

### Training

Train the model on your dataset:

```bash
# Use default configuration
python train.py --config-name conf_prototype

# Custom configuration
python train.py --config-name your_config \
    epochs=20 \
    batch_size=4 \
    lr=2e-4
```

**Training configuration** (`conf/conf_prototype.yaml`):
```yaml
epochs: 20
batch_size: 4
steps_per_epoch: 200
lr: 2e-4
loss: "mae"
```

## Datasets

The model is trained on a combination of clean audio datasets with synthetic noise:

| Dataset | Type | Usage | Files |
|---------|------|-------|-------|
| [AISHELL-3](https://www.openslr.org/93/) | Speech | Dialogue scenes | 963 train, ~150 val |
| [FSD50K](https://zenodo.org/record/4060432) | Sound effects | Ambient sounds | 36,846 train, 500 val |
| [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) | Music | Film scores | 320 train, ~50 val |
| [Gramophone Noise](https://www.audiolabs-erlangen.de/resources/MIR/2019-WASPAA-Noise) | Noise | Synthetic degradation | 2,427 files |

**Dataset mixture:** 60% speech + 20% sound effects + 20% music

### Dataset Configuration

Edit `conf/dset/dataset_prototype.yaml` to specify dataset paths:

```yaml
path_music_train:
  - "/path/to/AISHELL-3/train/wav"
  - "/path/to/FSD50K/dev_audio"
  - "/path/to/MusicNet/train_data"

path_music_validation:
  - "/path/to/AISHELL-3/test/wav/SSB****"
  - "/path/to/FSD50K/eval_audio"
  - "/path/to/MusicNet/test_data"

path_noise:
  - "/path/to/Gramophone_Record_Noise_Dataset"
```

## Model Architecture

**Multi-stage U-Net** with the following specifications:

- Parameters: 1,880,388 (7.17 MB)
- Stages: 2
- Depth: 3 layers per stage
- Time-frequency blocks: 1 (num_tfc)
- Input: STFT magnitude (win_size=2048, hop_size=512)
- Output: Denoised STFT magnitude

## Project Structure

```
denoising-historical-recordings-FILM/
├── train.py                      # Training script
├── inference.py                  # Inference script
├── trainer.py                    # Training loop implementation
├── dataset_loader.py             # Data loading and preprocessing
├── unet.py                       # U-Net model architecture
│
├── conf/                         # Configuration files (Hydra)
│   ├── conf_prototype.yaml       # Main config
│   ├── conf_fifth.yaml           # Experimental config
│   └── dset/
│       └── dataset.yaml          # Dataset paths
│
├── docs/                         # Documentation
│   ├── TRAINING_SUMMARY.md       # Training history and analysis
│   ├── M3_GPU_GUIDE.md          # Apple Silicon setup guide
│   └── WHY_LOSS_CONVERGED.md    # Problem analysis
│
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Known Issues

### Issue 1: Over-denoising (90% signal removal)

**Cause:** Mismatch between training noise (gramophone) and target noise (film)

**Workaround:** Use reduced denoising strength (0.2-0.5)

**Potential solutions:**
- Use film-specific noise for training
- Implement perceptual loss functions
- Apply unsupervised methods (e.g., Noise2Noise)

### Issue 2: MAE loss doesn't reflect perceptual quality

**Observation:** Model achieves low validation loss but poor subjective quality

**Recommendations:**
- Add multi-scale STFT loss
- Incorporate perceptual loss (VGGish)
- Use combined loss: `λ₁·MAE + λ₂·STFT + λ₃·Perceptual`

For detailed analysis, see [docs/WHY_LOSS_CONVERGED.md](docs/WHY_LOSS_CONVERGED.md)

## Performance Notes

- **Training speed**: ~4-6 seconds per step (varies by hardware)
- **Inference speed**: ~1.5-2 seconds per audio segment
- **Memory requirements**: 4-8 GB RAM for training (batch_size=4)
- **Storage**: Model checkpoint ~7 MB

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{moliner2022two,
  title={A two-stage U-Net for high-fidelity denoising of historical recordings},
  author={Moliner, Eloi and V{\"a}lim{\"a}ki, Vesa},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={841--845},
  year={2022},
  organization={IEEE}
}
```

## License

MIT License

**Original Work**: Copyright (c) 2021 Eloi Moliner Juanpere  
**This Adaptation**: Modified and extended under the same MIT License

See [LICENSE](LICENSE) for full terms.

## Acknowledgments

This project is built upon the following work:

### Original Research
- **Author**: Eloi Moliner Juanpere
- **Repository**: [denoising-historical-recordings](https://github.com/eloimoliner/denoising-historical-recordings)
- **Paper**: E. Moliner and V. Välimäki, "A two-stage U-Net for high-fidelity denoising of historical recordings", ICASSP 2022

### Datasets
- **AISHELL-3**: Chinese speech corpus for speech synthesis
- **FSD50K**: Freesound Dataset 50K for sound event classification
- **MusicNet**: Classical music dataset for note-level transcription
- **Gramophone Record Noise Dataset**: Historical recording noise samples

### Tools and Frameworks
- **TensorFlow**: Deep learning framework
- **Hydra**: Configuration management system

---

**Adaptation Note**: This project adapts the original gramophone record denoising model for film audio restoration. The core Two-stage U-Net architecture is preserved from the original work. Dataset integration, training pipeline modifications, and film-specific optimizations are contributions of this adaptation.


