# QFI-Diff: Quantum Fisher Information Guided Diffusion

Official PyTorch implementation of **"Quantum Fisher Information Guided Diffusion Refinement for Fuzzy and Uncertain Surveillance Regions"**. 

This framework utilizes a quantum-information-inspired reliability mask (QFI) computed via classical tensor-network contraction of a 2D Transverse-Field Ising Model. The QFI mask modulates the reverse SDE of a Latent Diffusion Model (LDM) to prevent forensic hallucinations in information-sparse regions.

## Hardware & Training Provenance
- **Hardware:** Single NVIDIA A100 (80GB)
- **Training Time:** ~14.2 GPU-hours
- **Optimizer:** AdamW (lr=$1\times10^{-5}$, min_lr=$1\times10^{-7}$, cosine decay)
- **Inference Latency:** 4.65 seconds/frame (50 DDIM steps)

## Baseline Reproducibility
To ensure strict traceability, baseline comparisons (SRGAN, SwinIR, DiffPIR, ResShift) are conducted using their official public pretrained checkpoints evaluated purely off-the-shelf, without degradation-specific retraining. See `run_baselines.sh` for exact inference protocols.
