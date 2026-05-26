# Deepfake-Detection

# Semantic-Conditioned Tri-Stream Fusion for Cross-Dataset Deepfake Detection

Official research code repository for:

> **Semantic-Conditioned Tri-Stream Fusion for Cross-Dataset Deepfake Detection**  
> Jong-Chan Park, Su-Jin Park, Yu-Jin Ha, Sang-Min Choi, and Gun-Woo Kim

This repository contains the experimental code for cross-dataset deepfake detection using complementary **RGB spatial**, **wavelet frequency**, and **semantic** representations.

The main objective of this work is to improve zero-shot generalization under distribution shifts caused by unseen manipulation methods, compression and re-encoding, resolution changes, and differences in capture pipelines.

---

## Overview

Deepfake detectors often achieve strong performance on the dataset used for training, but their performance can degrade substantially when evaluated on unseen datasets.

To address this problem, we propose **TriStreamFusion**, a semantic-conditioned tri-stream architecture that combines:

- **RGB spatial cues** for facial boundaries, skin texture, and blending artifacts.
- **Wavelet frequency cues** for multi-resolution forgery inconsistencies.
- **Semantic cues** extracted from a frozen CLIP image encoder.

The proposed fusion block treats RGB and wavelet representations as the main discriminative features and uses semantic representations as auxiliary information for cross-attention-based refinement.

---

## Proposed Architecture

The final architecture is composed of three streams:

| Stream | Backbone / Representation | Purpose |
|---|---|---|
| RGB Spatial Stream | ConvNeXtV2-Tiny with channel refinement | Captures boundary artifacts, abnormal textures, and blending traces |
| Wavelet Frequency Stream | ResNet-50 with Sym4 Level-2 SWT input | Captures multi-resolution frequency inconsistencies |
| Semantic Stream | Frozen CLIP ViT-B/32 | Provides content-level semantic guidance |

### Main-to-Aux Semantic-Conditioned Fusion

The proposed fusion strategy is designed as follows:

1. RGB and wavelet streams form the **main representation**.
2. Semantic features extracted from CLIP form the **auxiliary key-value source**.
3. Main features attend to semantic features through a cross-attention block.
4. The fused representation is passed to an MLP classification head for real/fake prediction.

In the final paper configuration, the semantic stream uses selected semantic tokens with:

```text
Ns = k = 5
```

This enables the RGB and wavelet main tokens to selectively attend to multiple semantic cues rather than relying on a single global representation.

---

## Wavelet Representation

The wavelet stream uses a **Sym4-based Level-2 Stationary Wavelet Transform (SWT)**.

For each RGB input frame:

1. Level-1 SWT produces:

```text
LL1, LH1, HL1, HH1
```

2. The low-frequency component `LL1` is decomposed again to obtain:

```text
LL2, LH2, HL2, HH2
```

3. A multi-resolution subband energy map is constructed from the high-frequency components.

4. The final wavelet representation combines:

```text
LL2 + Energy Map
```

This representation is intended to preserve structural information while emphasizing frequency inconsistencies caused by manipulation, resampling, blending, and compression.

---

## Cross-Dataset Evaluation Protocol

### Training Dataset

| Dataset | Usage |
|---|---|
| FaceForensics++ (FF++) | Source-domain training and validation |

### Zero-Shot Test Datasets

| Dataset | Usage |
|---|---|
| Celeb-DF v2 | External zero-shot evaluation |
| DFDC | External zero-shot evaluation |
| DeepfakeTIMIT | External zero-shot evaluation |
| WildDeepfake | External zero-shot evaluation |

The model is trained only on **FaceForensics++** and directly evaluated on unseen external datasets without target-domain fine-tuning or threshold calibration.

### Evaluation Details

- Input resolution: `224 × 224`
- Frame sampling interval: every 5 frames
- Frame-level fake probabilities are averaged within each video.
- All reported classification metrics are computed at the **video level**.
- Fixed prediction threshold: `0.5`
- Main experiments are repeated over five random seeds.

### Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-Binary
- ROC-AUC

Because the DeepfakeTIMIT evaluation subset contains only fake videos, pooled metrics are reported in two settings:

| Setting | Description |
|---|---|
| Pooled-all | Celeb-DF v2 + DFDC + DeepfakeTIMIT + WildDeepfake |
| Pooled-w/o-DeepfakeTIMIT | Celeb-DF v2 + DFDC + WildDeepfake |

---

## Main Results

### Video-Level Cross-Dataset Performance

| Model | Pooled-all Acc. | Pooled-all F1-B | Pooled-all AUC | Pooled-w/o-DeepfakeTIMIT Acc. | Pooled-w/o-DeepfakeTIMIT F1-B |
|---|---:|---:|---:|---:|---:|
| IDCNet | 57.52 | 64.86 | 65.49 | 59.64 | 66.82 |
| UCF | 68.22 | 77.16 | 64.48 | 67.05 | 75.93 |
| F3Net-FAD | 70.92 | 79.33 | **70.01** | 69.85 | 78.23 |
| Xception | 69.47 | 78.34 | 66.27 | 68.34 | 77.19 |
| EfficientNet-B7 | 69.80 | 79.37 | 68.63 | 68.69 | 78.31 |
| M2TR | 71.71 | 83.52 | 54.16 | 70.66 | 82.81 |
| **TriStreamFusion (Ours)** | **73.91** | **83.66** | 69.78 | **72.96** | **82.90** |

The proposed model achieves the highest pooled accuracy under both pooled evaluation settings while maintaining competitive fake-class F1 performance and a comparable pooled ROC-AUC.

---

## Repository Structure

```text
Deepfake-Detection/
├── RGBsparial_step1/
│   ├── Xception/
│   ├── resnet/
│   ├── efficientNet.py
│   ├── efficientNet_test.py
│   ├── hornet_test.py
│   ├── maxvit_test.py
│   ├── train.sh
│   └── test.sh
│
├── Frequency_step2/
│   ├── train.py
│   ├── train_wavelet.py
│   ├── train_wavelet2.py
│   ├── train_wavelet2_1.py
│   ├── train_dct_fft2.py
│   ├── test_wavelet2.py
│   ├── train_repeat.sh
│   └── test_repeat.sh
│
├── Dual_fusion_step/
│   ├── RGB_Wavelet/
│   └── RGB_Wavelet_DCT/
│
├── Tri_steam/
│   ├── train.py
│   ├── test.py
│   └── train_test_repeat.sh
│
├── comparison/
│   ├── F3Net/
│   ├── IDCNet/
│   ├── M2TR/
│   ├── UCF/
│   └── _result/
│
├── othre/
│   ├── Grad_CAM.py
│   ├── Grad_CAM_v2.py
│   ├── Grad_CAM_v3.py
│   ├── ensemble.py
│   ├── ensemble_mlp_2.py
│   ├── ensemble_mlp_3.py
│   ├── meta_ensemble.py
│   └── test_3stream_v2.py
│
├── tool/
│   ├── download_ffpp.py
│   ├── img_extractor.py
│   ├── img_mtcnn_ectract.py
│   ├── mtcnn_extract_face.py
│   ├── data_count.py
│   ├── avg_result.py
│   └── data_eda.ipynb
│
└── README.md
```

### Directory Description

| Directory | Description |
|---|---|
| `RGBsparial_step1/` | Single-stream RGB backbone experiments and evaluations |
| `Frequency_step2/` | Wavelet, DCT, FFT, and frequency-domain experiments |
| `Dual_fusion_step/` | Dual-stream fusion experiments |
| `Tri_steam/` | Tri-stream training and testing scripts |
| `comparison/` | Baseline model implementations and evaluation results |
| `othre/` | Grad-CAM visualization, ensemble experiments, and supplementary analysis |
| `tool/` | Dataset preprocessing, frame extraction, counting, and visualization utilities |

> Note: Several directory names reflect the original experimental workspace naming convention, such as `RGBsparial_step1`, `Tri_steam`, and `othre`.

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/park3010/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2. Create Environment

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
```

### 3. Install Required Packages

```bash
pip install torch torchvision
pip install numpy pandas opencv-python pillow tqdm
pip install scikit-learn pywavelets transformers
pip install facenet-pytorch
```

Additional dependencies may be required depending on the selected backbone or baseline implementation.

---

## Dataset Preparation

### FaceForensics++

FaceForensics++ is used as the source training dataset.

The expected training data structure is:

```text
<FFPP_ROOT>/
├── original_sequences/
│   └── <method>/
│       └── raw/
│           └── mtcnn/
└── manipulated_sequences/
    └── <method>/
        └── raw/
            └── mtcnn/
```

Preprocessing utilities for frame extraction and face alignment are provided in:

```text
tool/
├── download_ffpp.py
├── img_extractor.py
├── img_mtcnn_ectract.py
└── mtcnn_extract_face.py
```

### External Test Datasets

The following external datasets are used for zero-shot evaluation:

```text
Celeb-DF v2
DFDC
DeepfakeTIMIT
WildDeepfake
```

Please download each dataset from its official distribution source and follow its respective license and usage policy.

---

## Training

### RGB Spatial Stream

RGB-only experiments are implemented under:

```text
RGBsparial_step1/
```

Example:

```bash
cd RGBsparial_step1
bash train.sh
```

### Wavelet Frequency Stream

Wavelet and frequency-domain experiments are implemented under:

```text
Frequency_step2/
```

Example:

```bash
cd Frequency_step2
python train_wavelet2_1.py \
    --data-dir <FFPP_ROOT> \
    --compression raw
```

### Tri-Stream Fusion Model

The tri-stream model is implemented under:

```text
Tri_steam/
```

Example command template:

```bash
cd Tri_steam

python train.py \
    --data-dir <FFPP_ROOT> \
    --compression raw \
    --img-size 224 \
    --streams rgb,wavelet,semantic \
    --gpu 0
```

Depending on the experiment configuration, branch checkpoints may be supplied for RGB and wavelet streams before fusion training.

---

## Evaluation

### Tri-Stream Testing

```bash
cd Tri_steam

python test.py \
    --checkpoint <CHECKPOINT_PATH> \
    --data-dir <TEST_DATA_ROOT> \
    --gpu 0
```

### Repeated Experiments

Repeated training and testing scripts are provided for multi-seed evaluation:

```bash
cd Tri_steam
bash train_test_repeat.sh
```

The paper reports results averaged over five repeated experiments with different random seeds.

---

## Baseline Comparison

Representative comparison methods are organized under:

```text
comparison/
├── F3Net/
├── IDCNet/
├── M2TR/
└── UCF/
```

The comparison protocol follows the same overall evaluation setting:

- FaceForensics++ as the source training domain
- External zero-shot test datasets
- Input resolution of `224 × 224`
- Video-level aggregation of frame predictions
- Fixed threshold of `0.5`

---

## Visualization and Additional Analysis

The repository includes supplementary analysis scripts for interpreting model behavior.

### Grad-CAM

Grad-CAM-related scripts are located in:

```text
othre/
├── Grad_CAM.py
├── Grad_CAM_v2.py
├── Grad_CAM_v3.py
└── grad_cam.sh
```

### Ensemble Analysis

Ensemble and meta-classification experiments are located in:

```text
othre/
├── ensemble.py
├── ensemble_mlp_2.py
├── ensemble_mlp_3.py
├── meta_ensemble.py
└── triple_schreme_mlp.py
```

---

## Important Notes

- The repository currently reflects the experimental workspace used during model development and ablation studies.
- Paths for datasets, checkpoints, and result directories may need to be configured manually in each script or shell file.
- Before running the final paper configuration, please verify that the tri-stream implementation matches the final reported architecture, particularly the semantic token configuration used in semantic-conditioned fusion.
- Trained checkpoints and preprocessed datasets are not included unless explicitly provided.

---

## Citation

The paper citation information will be updated after publication.

```bibtex
@article{park2026tristreamfusion,
  title   = {Semantic-Conditioned Tri-Stream Fusion for Cross-Dataset Deepfake Detection},
  author  = {Park, Jong-Chan and Park, Su-Jin and Ha, Yu-Jin and Choi, Sang-Min and Kim, Gun-Woo},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Publication information to be updated}
}
```

---

## License

The license for this repository will be specified in a future update.

Third-party datasets and baseline implementations remain subject to their original licenses and usage policies.

---

## Contact

For questions regarding this work, please contact the authors through the corresponding author information provided in the paper.
