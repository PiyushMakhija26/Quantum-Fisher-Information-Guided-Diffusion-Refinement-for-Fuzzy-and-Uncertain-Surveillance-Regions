# QFI-Diff: Quantum Fisher Information Guided Diffusion Refinement for Fuzzy and Uncertain Surveillance Regions

*Official Repository Documentation*

---

## Overview

Welcome to the official repository for **QFI-Diff**. This project introduces a novel, quantum-information-inspired framework designed to solve the critical challenges of image restoration in highly degraded, real-world surveillance environments.

By integrating principles from Quantum Metrology with modern Latent Diffusion Models (LDMs) executed entirely on classical hardware, QFI-Diff bridges the gap between generative perceptual quality and strict forensic structural integrity.

---

## The Problem: The Surveillance Fidelity Paradox

Restoring heterogeneous surveillance imagery—such as grainy night-vision feeds or hazy perimeter cameras—presents a unique paradox for modern Artificial Intelligence:

- **Conventional Denoising (Regression Models):** Traditional AI methods optimize for the average of all possible textural solutions, resulting in severe texture wiping and blurry images that destroy high-frequency forensic details.
- **Generative AI (Diffusion Models):** While capable of synthesizing highly realistic textures, unconstrained diffusion models act as "dreaming machines." In regions of heavy sensor noise or shadows, they hallucinate statistically probable but factually incorrect features, violating the strict admissibility standards required in forensic analysis.

---

## Our Methodology: The QFI-Diff Framework

To solve this paradox, our framework explicitly separates the quantification of local visual uncertainty from the generative restoration process. It operates in two synergistic streams:

### 1. The Quantum Uncertainty Stream
We require a hyper-sensitive sensor capable of distinguishing between structurally invariant stochastic sensor noise and genuine semantic edges. To achieve this, we turn to Quantum Information Theory.

- **State Encoding:** We map classical image patches into probability amplitudes, transforming visual data into a robust quantum state representation.
- **Structural Probing:** We simulate a Transverse-Field Ising Model using classical Tensor Network approximations. By perturbing the encoded visual state with this physical model, we measure its geometric sensitivity.
- **The Reliability Mask:** The resulting variance of this measurement yields the Quantum Fisher Information (QFI). This metric generates a highly precise, deterministic spatial reliability map that dictates exactly where the image contains verified structural evidence and where it has decayed into pure noise.

### 2. The Classical Generative Stream
With the QFI reliability mask generated, we guide a pre-trained Latent Diffusion Model to reconstruct the image.

- **Manifold Consistency Constraint:** During the reverse diffusion sampling process, the QFI mask acts as a dynamic constraint.
- **Evidence Preservation:** In highly reliable regions (clear geometry), the mask forces the diffusion model to strictly anchor to the original noisy observation, actively preventing the network from hallucinating false details.
- **Generative Refinement:** In highly uncertain regions (fuzzy shadows), the constraint is lifted, freeing the generative prior to synthesize highly plausible, artifact-free textures to resolve the visual ambiguity.

---

## Repository Structure

This repository contains the complete codebase required to reproduce the QFI-Diff pipeline:

- **QFI Core:** Modules for extracting the QFI reliability mask using Tensor Network contractions.
- **Diffusion Pipeline:** Scripts for fine-tuning the autoencoder domain adaptation and executing the QFI-guided DDIM sampling loop.
- **Evaluation Suite:** A comprehensive set of tools for calculating structural fidelity, perceptual realism, and downstream object detection accuracy, alongside robust statistical significance testing.

---

## Forensic Interpretation and Operational Limits

While QFI-Diff achieves state-of-the-art balance between structural retention and perceptual quality, we explicitly constrain our claims to a *partial information regime*. The framework does not magically create physical information that is completely absent from the observation. In cases of total occlusion (e.g., pitch-black shadows with zero photon data), the model defaults to its generative prior. Such generations represent a model-based hypothesis rather than recovered evidence and must remain strictly flagged as legally unverified by forensic analysts.

---

## Hardware and Reproducibility

The primary benchmarking and fine-tuning phases for this research were accelerated using an NVIDIA A100 GPU. However, to ensure broad accessibility for forensic laboratories, the framework's memory footprint is heavily optimized, allowing the entire inference pipeline to be successfully executed on standard consumer-grade hardware.

To guarantee fair comparative benchmarking, all baseline models evaluated in our study utilize their official, publicly available pre-trained checkpoints evaluated strictly off-the-shelf.
