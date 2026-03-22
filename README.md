# 🍼 Infant State Recognition System

> Classifying infant cry states using classical ML and deep learning — built for accuracy, interpretability, and real-world deployment.

**AML Project 3 | Domain: Healthcare · Assistive Technology · Consumer IoT**

---

## What This Project Does

A baby cries — but why? This project builds an audio classification pipeline that listens to a 3-second cry clip and identifies one of five states:

| State | Description |
|---|---|
| 😢 Hungry | Rhythmic, medium-pitched cry |
| 😣 Pain | High-pitched, harsh, urgent |
| 😤 Discomfort | Medium pitch, moderate noise |
| 🫧 Burping | Low pitch, fast rhythm, clean |
| 😴 Tired | Low-medium pitch, gentle rhythm |

---

## Models Built

| Track | Model | Accuracy | F1 |
|---|---|---|---|
| Baseline ML | Decision Tree | 98.5% | 0.985 |
| Advanced ML | Random Forest (GridSearchCV) | 99.5% | 0.995 |
| Advanced ML | SVM with RBF Kernel | 99.0% | 0.990 |
| Deep Learning | MelCNN (CNN on spectrograms) | — | — |
| Hybrid/Edge | ExtraTrees (28 features, 143KB) | 100% | 1.000 |

> The CNN underperforms on synthetic data by design — see the notebook for the full failure analysis and what it tells us about model complexity vs data complexity.

---

## Project Structure

```
infant-cry-classification/
│
├── Infant_Cry_Classification.ipynb   ← Main notebook (run this)
├── README.md
│
├── reports/                          ← All generated plots
│   ├── eda_overview.png
│   ├── spectrograms.png
│   ├── tsne.png
│   ├── decision_tree.png
│   ├── random_forest.png
│   ├── cnn_training.png
│   ├── edge_results.png
│   └── final_comparison.png
│
└── models/                           ← Saved trained models
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── svm.pkl
    ├── edge_model.pkl
    └── mel_cnn.keras
```

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open the notebook in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Runtime → **Run All**
4. Takes ~15 minutes. All plots and models download automatically at the end.

### Option 2: Local
```bash
pip install librosa soundfile scikit-learn tensorflow matplotlib seaborn joblib
jupyter notebook Infant_Cry_Classification.ipynb
```

---

## How It Works

### Audio → Features (Classical ML path)
Each 3-second clip at 8 kHz is converted into an **88-number feature vector**:

- **MFCCs** (40 dims) — describe the shape of the vocal tract at each moment
- **Delta MFCCs** (20 dims) — how fast the sound is changing
- **Delta-Delta MFCCs** (20 dims) — acceleration of change
- **Spectral features** (8 dims) — brightness, roughness, loudness

> Why 8 kHz? Infant cry fundamentals are 300–800 Hz. The Nyquist theorem tells us 8 kHz (Nyquist = 4 kHz) captures everything we need — at 2.75× less compute than the standard 22 kHz.

### Audio → Spectrogram (Deep Learning path)
Each clip is converted to a **64 × 188 mel-spectrogram image** — then a CNN learns patterns directly from the image.

---

## Key Findings

**1. Classical ML is surprisingly competitive**
Random Forest achieves 99.5% F1 with just 88 handcrafted features. The features themselves carry most of the discriminative power.

**2. The CNN failure is informative, not a mistake**
The CNN severely overfits on synthetic data because synthetic audio is linearly separable — classical models can solve it with a handful of rules. The CNN is designed for real acoustic variability. On real donateacry recordings, it would be expected to match or outperform classical models.

**3. Edge deployment is feasible**
The ExtraTrees model fits in **143 KB**, runs in **57ms average** on CPU, and maintains **91.9% F1** even under heavy noise — meeting all real-time IoT constraints.

---

## Dataset

We use a synthetic dataset that mirrors the [donateacry-corpus](https://github.com/gveres/donateacry-corpus) structure:
- 5 classes × 200 clips = **1,000 audio files**
- Each clip: 3 seconds, 22,050 Hz, mono WAV

To use real data, replace `data/raw/` with the donateacry corpus folder structure and re-run.

---

## Dependencies

```
librosa >= 0.10
soundfile
scikit-learn >= 1.3
tensorflow >= 2.13
matplotlib
seaborn
joblib
numpy
```

## Authors

Aman Kumar · AML Project 3
