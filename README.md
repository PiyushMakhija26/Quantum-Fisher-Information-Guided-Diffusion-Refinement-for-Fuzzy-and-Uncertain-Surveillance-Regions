# QFI-Diff: Quantum Fisher Information Guided Diffusion Refinement for Fuzzy and Uncertain Surveillance Regions

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.20+-orange.svg?style=flat&logo=huggingface)](https://github.com/huggingface/diffusers)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_A100_(80GB)-green.svg)](#hardware--training-provenance)

Official PyTorch implementation of the research paper:  
**"Quantum Fisher Information Guided Diffusion Refinement for Fuzzy and Uncertain Surveillance Regions"**.

---

## Table of Contents
- [1. Overview & Methodological Motivation](#1-overview--methodological-motivation)
- [2. Theoretical Foundations](#2-theoretical-foundations)
  - [2.1 Quantum State Encoding from Local Patches](#21-quantum-state-encoding-from-local-patches)
  - [2.2 2D Transverse-Field Ising Model (TFIM)](#22-2d-transverse-field-ising-model-tfim)
  - [2.3 Quantum Fisher Information (QFI) Extraction](#23-quantum-fisher-information-qfi-extraction)
  - [2.4 QFI Reliability Mask Derivation](#24-qfi-reliability-mask-derivation)
  - [2.5 Latent DDIM Reverse SDE with Manifold Consistency](#25-latent-ddim-reverse-sde-with-manifold-consistency)
- [3. Repository Architecture](#3-repository-architecture)
- [4. Installation & Environment Setup](#4-installation--environment-setup)
- [5. Dataset Preparation](#5-dataset-preparation)
- [6. Pipeline Walkthrough & Execution Guide](#6-pipeline-walkthrough--execution-guide)
  - [Stage 1: VQ-GAN Domain Adaptation](#stage-1-vq-gan-domain-adaptation)
  - [Stage 2: Core QFI-Diff Inference](#stage-2-core-qfi-diff-inference)
  - [Stage 3: Multi-Seed Statistical Validation Protocol](#stage-3-multi-seed-statistical-validation-protocol)
  - [Stage 4: Downstream Face Detection (YOLOv5)](#stage-4-downstream-face-detection-yolov5)
- [7. Baseline Reproducibility](#7-baseline-reproducibility)
- [8. Hardware & Training Provenance](#8-hardware--training-provenance)
- [9. Citation](#9-citation)

---

## 1. Overview & Methodological Motivation

In unconstrained surveillance scenarios (e.g., severe underexposure, motion blur, sensor noise, atmospheric haze), standard deep generative models and Latent Diffusion Models (LDMs) suffer from **forensic hallucination**: in regions devoid of physical information, standard unconstrained diffusion draws samples from its unconditioned prior, fabricating plausibly realistic but forensically unfaithful facial features and textural artifacts.

**QFI-Diff** eliminates forensic hallucinations by computing a deterministic quantum-mechanical reliability mask directly from observation patches:
1. **Physical Uncertainty Quantization:** Pixel patches are mapped onto pure product states of an interconnected qubit lattice governed by a 2D Transverse-Field Ising Model (TFIM).
2. **Quantum Fisher Information (QFI):** The quantum parameter sensitivity of the Hamiltonian variance is evaluated per patch. High QFI corresponds to structural edge transitions and information-rich boundaries, whereas low QFI indicates degenerate, information-sparse, noise-dominated regions.
3. **Manifold Consistency Fusion:** The normalized QFI mask dynamically blends the DDIM reverse diffusion trajectory with the degraded observation latent, allowing generative prior completion in uncertain regions while strictly preserving observation fidelity where real signal exists.

```
Degraded Input Observation (Y)
          │
          ├───► [ QFI Extractor (2D TFIM) ] ──────────► Reliability Mask M(x,y)
          │                                                   │
          └───► [ VQ-GAN Encoder ] ──► z_0_obs                ▼
                                         │            [ Latent Pool (8x) ]
                                         │                    │
                                         ▼                    ▼
                                [ Reverse DDIM Loop (t = T -> 0) ]
                                  z_{t-1} = (1 - M_z) * z_y + M_z * z_pred
                                         │
                                         ▼
                                [ Finetuned VQ-GAN Decoder ]
                                         │
                                         ▼
                             Forensically Restored Image
```

---

## 2. Theoretical Foundations

### 2.1 Quantum State Encoding from Local Patches
Given an input grayscale observation normalized to $r \in [0, 1]^{H \times W}$, non-overlapping local patches of dimension $k \times k$ ($N = k^2$, default $k=4 \implies N=16$ qubits) are extracted. Each normalized pixel intensity $r_a$ maps to a single-qubit state in Hilbert space $\mathbb{C}^2$:
$$|\phi_a\rangle = \sqrt{1 - r_a}|0\rangle + \sqrt{r_a}|1\rangle$$

The composite patch quantum state $|\Psi\rangle \in \mathcal{H}^{\otimes N}$ of dimension $2^N$ is formed via tensor product:
$$|\Psi\rangle = \bigotimes_{a=1}^N |\phi_a\rangle$$

### 2.2 2D Transverse-Field Ising Model (TFIM)
The patch lattice is governed by a 2D Transverse-Field Ising Hamiltonian $\mathcal{H}$:
$$\mathcal{H} = -h \sum_{i=1}^N \sigma_i^x - J \sum_{\langle i, j \rangle} \sigma_i^z \sigma_j^z$$
where:
- $\sigma_i^x$ and $\sigma_i^z$ denote the Pauli spin matrices acting on qubit $i$.
- $\langle i, j \rangle$ denotes nearest-neighbor interactions on the 2D planar grid (horizontal and vertical lattice edges).
- $h$ is the transverse field parameter ($h = 0.5$).
- $J$ is the ferromagnetic coupling coefficient ($J = 1.0$), enforcing a ratio $J/h = 2.0$.

### 2.3 Quantum Fisher Information (QFI) Extraction
For a pure quantum state $|\Psi\rangle$ parameterized by Hamiltonian dynamics, the Quantum Fisher Information $\mathcal{F}_Q$ with respect to the state evolution generator is proportional to the quantum variance of $\mathcal{H}$:
$$\mathcal{F}_Q = 4 \operatorname{Var}_\Psi(\mathcal{H}) = 4 \left( \langle \Psi | \mathcal{H}^2 | \Psi \rangle - \left( \langle \Psi | \mathcal{H} | \Psi \rangle \right)^2 \right)$$

This metric directly captures quantum criticality and the local susceptibility of the observed lattice patch to phase transitions.

### 2.4 QFI Reliability Mask Derivation
The patch-wise raw QFI matrix is spatially interpolated back to the original image dimensions $(H, W)$. Let $\mu_{\mathcal{F}}$ and $\sigma_{\mathcal{F}}$ denote the empirical mean and standard deviation of $\mathcal{F}_Q$ across the image. The normalized reliability mask $M \in (0, 1)^{H \times W}$ is computed using a temperature-scaled logistic transformation:
$$M(x, y) = \frac{1}{1 + \exp\left( \frac{\mathcal{F}_Q(x, y) - \mu_{\mathcal{F}}}{\gamma \cdot \sigma_{\mathcal{F}}} \right)}$$
with empirical scaling factor $\gamma = 1.0$.

### 2.5 Latent DDIM Reverse SDE with Manifold Consistency
In the latent space of the AutoencoderKL ($8\times$ spatial downsampling), the mask is downsampled via average pooling:
$$M_z = \operatorname{AvgPool2D}_{8\times8}(M)$$

At each reverse DDIM timestep $t \to t-1$, given model noise prediction $\epsilon_\theta(z_t, t)$, the standard clean latent estimate $\hat{z}_0$ and forward step $z_{t-1}^{\text{pred}}$ are:
$$\hat{z}_0 = \frac{z_t - \sqrt{1 - \alpha_t} \epsilon_\theta(z_t, t)}{\sqrt{\alpha_t}}$$
$$z_{t-1}^{\text{pred}} = \sqrt{\alpha_{t-1}} \hat{z}_0 + \sqrt{1 - \alpha_{t-1}} \epsilon_\theta(z_t, t)$$

Simultaneously, the forward observation trajectory perturbed to timestep $t-1$ is:
$$z_{t-1}^y = \sqrt{\alpha_{t-1}} z_0^{\text{obs}} + \sqrt{1 - \alpha_{t-1}} \eta, \quad \eta \sim \mathcal{N}(0, \mathbf{I})$$

The final manifold-consistent update blends both representations via $M_z$:
$$z_{t-1}^{\text{fused}} = (1 - M_z) \odot z_{t-1}^y + M_z \odot z_{t-1}^{\text{pred}}$$

---

## 3. Repository Architecture

```text
QFI-Diff/
├── README.md                          # Comprehensive documentation & mathematical specification
├── requirements.txt                   # Dependency manifests (torch, diffusers, pyiqa, etc.)
├── run_baselines.sh                   # Off-the-shelf baseline inference execution script
├── run_statistical_eval.sh            # 5-seed validation & Wilcoxon significance testing pipeline
│
├── qfi_core/
│   └── qfi_extractor.py               # 2D TFIM Hamiltonian construction, state encoding & QFI mask generator
│
├── pipeline/
│   ├── vqgan_finetuner.py             # Domain-specific fine-tuning for AutoencoderKL decoder on DarkFace
│   ├── ddim_sampler.py                # Latent DDIM reverse step with QFI manifold consistency fusion
│   └── inference_qfidiff.py           # End-to-end inference driver executing 50-step restoration
│
└── evaluation/
    ├── evaluate_metrics.py            # Quantitative metric evaluation (PSNR, SSIM, LPIPS, FID, Ma, NIQE, PI)
    ├── statistical_tests.py           # Wilcoxon signed-rank test with Holm-Bonferroni correction
    └── yolov5_eval.py                 # Downstream facial detection validation using YOLOv5s
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Linux / Windows OS with CUDA 11.8 or 12.1+
- Python 3.9, 3.10, or 3.11
- NVIDIA GPU with $\ge$ 16GB VRAM (Single A100 80GB recommended for reproduction)

### Step-by-Step Environment Setup

```bash
# Clone the repository
git clone https://github.com/PiyushMakhija26/Quantum-Fisher-Information-Guided-Diffusion-Refinement-for-Fuzzy-and-Uncertain-Surveillance-Regions.git
cd Quantum-Fisher-Information-Guided-Diffusion-Refinement-for-Fuzzy-and-Uncertain-Surveillance-Regions

# Create and activate virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

The evaluation protocol benchmarked in the paper focuses on extremely low-light and degraded surveillance conditions (DarkFace).

Organize the dataset under `./data/` as follows:

```text
data/
├── darkface_train/
│   └── unpaired_5000/                 # 5,000 unpaired surveillance images for VQ-GAN fine-tuning
└── darkface_test/
    ├── degraded/                      # Low-light, noisy input observations
    ├── clean/                         # Ground-truth reference frames
    └── labels/                        # YOLO format bounding box annotations
```

---

## 6. Pipeline Walkthrough & Execution Guide

### Stage 1: VQ-GAN Domain Adaptation
Fine-tune the AutoencoderKL decoder to mitigate out-of-distribution blur and color shifting on low-light surveillance data:

```bash
python pipeline/vqgan_finetuner.py \
    --data_dir ./data/darkface_train/unpaired_5000 \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --min_lr 1e-7 \
    --weight_decay 1e-4 \
    --epochs 50 \
    --device cuda
```

### Stage 2: Core QFI-Diff Inference
Run single-seed restoration combining the deterministic QFI mask with DDIM 50-step reverse diffusion:

```bash
python pipeline/inference_qfidiff.py \
    --input_dir ./data/darkface_test/degraded/ \
    --output_dir ./outputs/qfidiff_restored/ \
    --vqgan_weights ./checkpoints/vqgan_finetuned_epoch50.pth \
    --seed 42 \
    --steps 50 \
    --device cuda
```

### Stage 3: Multi-Seed Statistical Validation Protocol
To reproduce the 5-seed distribution results, per-image metric variance, and the Holm-Bonferroni corrected Wilcoxon signed-rank test against baseline distributions:

```bash
# Ensure execution permissions on Unix environments
chmod +x run_statistical_eval.sh
./run_statistical_eval.sh
```

Or execute manually:
```bash
# 1. Run 5 independent seeds
for SEED in 42 123 456 789 999; do
    python pipeline/inference_qfidiff.py \
        --input_dir ./data/darkface_test/degraded/ \
        --output_dir ./outputs/qfidiff_seed_$SEED/ \
        --seed $SEED \
        --vqgan_weights ./checkpoints/vqgan_finetuned_epoch50.pth
        
    python evaluation/evaluate_metrics.py \
        --pred_dir ./outputs/qfidiff_seed_$SEED/ \
        --gt_dir ./data/darkface_test/clean/
done

# 2. Compute Wilcoxon Signed-Rank Test with Holm-Bonferroni correction
python evaluation/statistical_tests.py \
    --qfi_scores ./outputs/qfidiff_seed_42/ssim_scores.json \
    --baseline_scores ./outputs/baselines/diffpir/ssim_scores.json ./outputs/baselines/resshift/ssim_scores.json
```

### Stage 4: Downstream Face Detection (YOLOv5)
Evaluate whether restored images improve downstream forensic utility (face detection mAP@0.5:0.95):

```bash
python evaluation/yolov5_eval.py \
    --pred_dir ./outputs/qfidiff_seed_42/ \
    --labels_dir ./data/darkface_test/labels/
```

---

## 7. Baseline Reproducibility

To guarantee zero data-leakage and fair off-the-shelf comparison, all baseline models (SwinIR, DiffPIR, ResShift) are evaluated using their official public pretrained checkpoints without task-specific retraining.

Run the end-to-end baseline evaluation script:

```bash
chmod +x run_baselines.sh
./run_baselines.sh
```

Individual baseline repositories invoked:
- **SwinIR:** [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) (Classical SR $\times 2$)
- **DiffPIR:** [yuanjiali/DiffPIR](https://github.com/yuanjiali/DiffPIR) (Deblur / Restoration prior)
- **ResShift:** [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift) (Real-world Super-Resolution)

---

## 8. Hardware & Training Provenance

| Parameter | Specification |
|:---|:---|
| **Compute Hardware** | $1 \times$ NVIDIA A100-SXM4 (80GB VRAM) |
| **CUDA / Driver** | CUDA 12.1 / Driver 535.104.05 |
| **Total Training Time** | ~14.2 GPU-hours (VQ-GAN Domain Adaptation, 50 epochs) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1\times 10^{-4}$) |
| **Learning Rate Schedule** | Cosine Annealing ($lr_{\max} = 1\times 10^{-5}, lr_{\min} = 1\times 10^{-7}$) |
| **Inference Latency** | $4.65\text{ s} \pm 0.12\text{ s}$ per $512\times 512$ frame (50 DDIM steps) |
| **Base Latent Diffusion** | Stable Diffusion 2.1 Base (`stabilityai/stable-diffusion-2-1-base`) |

---

