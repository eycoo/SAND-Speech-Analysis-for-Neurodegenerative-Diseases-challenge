# SAND-Speech-Analysis-for-Neurodegenerative-Diseases-challenge
AN IEEE ICASSP 2026 GRAND CHALLENGE This challenge stems from the need to analyse noninvasive, objective, and scalable biomarkers, such as speech signals, for early diagnosis and longitudinal monitoring of patients suffering from neurodegenerative diseases.

## ConvNeXt-Based Deep Learning Framework for ALS Dysarthria Severity Classification

> ICASSP 2026 SAND Challenge - Automated severity assessment of dysarthria in Amyotrophic Lateral Sclerosis patients using deep learning

## Overview

This repository presents a deep learning framework for automated classification of dysarthria severity in ALS patients. Our approach combines task-specific mel-spectrogram preprocessing with ConvNeXt-Base architecture, achieving **59.77% macro-averaged F1 score** on the ICASSP 2026 SAND Challenge validation set.

### Key Highlights

- **Efficient Architecture**: ConvNeXt-Base (89M parameters) achieves competitive performance approaching Vision Transformer baseline (60.6% F1)
- **Task-Specific Preprocessing**: Optimized mel-spectrogram extraction with task-dependent frequency ranges
- **Transfer Learning**: Leverages ImageNet-1K pretrained weights for accelerated convergence
- **Multi-Task Aggregation**: Patient-level predictions via majority voting across 8 speech tasks

## Problem Statement

Amyotrophic Lateral Sclerosis (ALS) is a progressive neurodegenerative disease that severely affects speech production through dysarthria. This work addresses automated severity assessment through 5-class classification:

- **Class 1**: Severe dysarthria
- **Class 2**: Moderately-severe
- **Class 3**: Moderate
- **Class 4**: Mild-moderate
- **Class 5**: Mild dysarthria

### Challenges

1. **Severe class imbalance**: Class 1 (2.2%) vs Class 5 (35.6%)
2. **Subtle acoustic differences** between adjacent severity levels
3. **Limited training samples**: 247 training patients, 53 validation patients

## Dataset

The dataset comprises speech recordings from 300 ALS patients performing 8 speech tasks:

**Phonation Tasks** (5 tasks)
- Sustained vowels: /a/, /e/, /i/, /o/, /u/

**Rhythm Tasks** (3 tasks)
- Syllable repetitions: /pa/, /ta/, /ka/

### Class Distribution

| Class | Percentage | Description |
|-------|-----------|-------------|
| 1 | 2.2% | Severe |
| 2 | 9.6% | Moderately-severe |
| 3 | 19.8% | Moderate |
| 4 | 32.8% | Mild-moderate |
| 5 | 35.6% | Mild |

## Methodology

### Preprocessing Pipeline

#### 1. Voice Activity Detection
- Energy-based VAD with 30 dB threshold
- RMS energy calculation (2048-sample frames, 512-sample hop)

#### 2. Task-Specific Frequency Optimization
Based on fundamental frequency (F0) analysis:
- **Phonation tasks**: 50-4000 Hz (captures F0 and harmonics)
- **Rhythm tasks**: 100-6000 Hz (preserves plosive characteristics)

#### 3. Mel-Spectrogram Extraction
- Sampling rate: 22.05 kHz
- Mel bins: 256
- FFT size: 2048
- Hop length: 512
- **Delta-delta features**: 3-channel stacking [mel, delta, delta²] for temporal dynamics

#### 4. Data Augmentation
- **Class oversampling**: Class 1 (21×), Class 2 (5×), Class 3 (2×)
- **SpecAugment**: Applied to 70% of samples
- **Image transforms**: Random flips, rotation, color jitter

### Model Architecture

**ConvNeXt-Base**
- 4 stages with depths [3, 3, 27, 3]
- Dimensions [128, 256, 512, 1024]
- Total parameters: 89M
- Key components:
  - Depthwise 7×7 convolution
  - LayerNorm
  - GELU activation
  - Inverted bottleneck design
  - Stochastic depth (0.1)

### Training Configuration

- **Loss function**: Focal loss (γ=2.0) with class weighting
- **Optimizer**: AdamW (lr=1e-4, weight decay=0.01)
- **Scheduler**: Cosine annealing
- **Epochs**: 75
- **Batch size**: 16
- **Precision**: Mixed precision training
- **Transfer learning**: ImageNet-1K pretrained weights

### Patient-Level Prediction

Multi-task aggregation via majority voting across all 8 speech tasks for improved robustness.

## Results

### Model Performance Comparison

| Model | Parameters | F1 Score | Accuracy |
|-------|-----------|----------|----------|
| ViT-Base (Baseline) | 86M | 0.606 | 0.42 |
| ConvNeXt-Small | 50M | 0.578 | 0.36 |
| **ConvNeXt-Base** | **89M** | **0.598** | **0.43** |
| ConvNeXt-Large | 198M | 0.598 | 0.53 |

### Per-Class Performance (ConvNeXt-Base)

| Class | Precision | Recall | F1 Score | Sample Size |
|-------|-----------|--------|----------|-------------|
| 1 (Severe) | 1.00 | 0.50 | 0.67 | 2 |
| 2 | 0.43 | 0.75 | 0.55 | 4 |
| 3 | 0.62 | 0.67 | 0.64 | 12 |
| 4 | 0.36 | 0.57 | 0.44 | 14 |
| 5 (Mild) | 0.80 | 0.38 | 0.52 | 21 |
| **Macro Average** | **0.64** | **0.57** | **0.56** | **53** |

### Key Observations

- **Primary confusion**: Class 5 → Class 4 (8/21 patients, 50% of errors)
- **Challenge**: Subtle acoustic differences between mild and moderate severity
- **Strength**: Perfect precision on severe cases (Class 1)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/als-dysarthria-classification.git
cd als-dysarthria-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Preprocessing

```bash
python preprocess.py --input_dir data/raw --output_dir data/processed
```

### Training

```bash
python train.py --config configs/convnext_base.yaml
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir data/processed
```

### Inference

```bash
python inference.py --audio_path path/to/audio.wav --model_path checkpoints/best_model.pth
```

## Future Work

To achieve the 70% baseline F1 score, we propose:

1. **Model Ensemble** (+3-5% F1): Combine ConvNeXt variants with different architectures
2. **Advanced Augmentation** (+2-3% F1): Mixup and CutMix for boundary examples
3. **Label Smoothing** (+1-2% F1): Soften hard class boundaries
4. **Test-Time Augmentation** (+1-2% F1): Multiple augmented inference passes
5. **Task-Specific Models** (+1-2% F1): Separate models for phonation and rhythm tasks

**Projected improvement**: +8-14% F1 (potential range: 68-74%)

## Project Structure

```
als-dysarthria-classification/
├── configs/                # Configuration files
├── data/                   # Dataset directory
├── models/                 # Model architectures
├── utils/                  # Utility functions
├── preprocess.py          # Preprocessing pipeline
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchaudio
- librosa
- numpy
- scikit-learn
- timm (PyTorch Image Models)

See `requirements.txt` for complete dependencies.

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{mboe2026convnext,
  title={ConvNeXt-Based Deep Learning Framework for ALS Dysarthria Severity Classification},
  author={Mboe, Jeremy Mattathias and Farras, Ahmad Naufal and Nikmah, Luna Arafatul},
  booktitle={ICASSP 2026 SAND Challenge},
  year={2026},
  organization={Institut Teknologi Sepuluh Nopember}
}
```

## Authors

**Jeremy Mattathias Mboe** - Department of Informatics Engineering, ITS  
**Ahmad Naufal Farras** - Department of Informatics Engineering, ITS  
**Luna Arafatul Nikmah** - Department of Medical Technology, ITS

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ICASSP 2026 SAND Challenge organizers
- Institut Teknologi Sepuluh Nopember
- ImageNet dataset and pretrained models
- ConvNeXt architecture by Liu et al.

---

**Note**: This project is part of the ICASSP 2026 SAND Challenge for automated assessment of dysarthria severity in ALS patients.
