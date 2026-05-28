# SSFA-HSI: Spectral-Spatial-Frequency Adaptation for Hyperspectral Image Coding for Machines

SSFA-HSI is a parameter-efficient, plug-and-play adapter framework designed specifically to bridge the dimensionality, spectral redundancy, and correlation challenges of Hyperspectral Imaging (HSI) within the **Image Coding for Machines (ICM)** and **Image Compression for Machine and Human Vision (ICMH)** paradigms.

This repository implements sequential adapters (**SpMA**, **SMA**, and **FMA**) embedded inside a pre-trained, frozen hierarchical VAE model (`qarv_hsi_lower`). It performs **Co-Adaptation** of the lightweight adapters alongside a downstream land-cover semantic segmentation head (U-Net) under a unified **Rate-Distortion-Task (RDT)** joint loss (MSE + Spectral Angle Mapper - SAM + Cross Entropy).

---

## 🚀 RTX 4090 GPU Server Deployment Guide

### 1. Environment Prerequisites

We recommend using **Anaconda** or **Miniconda** to manage python dependencies. Create and configure your environment as follows:

```bash
# 1. Create a clean Conda environment with Python 3.8
conda create -n ssfa_hsi python=3.8 -y
conda activate ssfa_hsi

# 2. Install PyTorch with CUDA 11.8+ support (optimized for RTX 4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install core dependencies
pip install timm tqdm scikit-image numpy
```

---

### 2. Dataset Preparation

This pipeline utilizes the **HySpecNet-11k** 8-bit dataset (11,483 patches of size $128 \times 128$ with 202 spectral bands).

1. **Locate your 8-bit HSI `.npy` files** on the server (e.g., `/path/to/hyspecnet11k/data`).
2. **Configure Dataset Paths** inside `lvae/paths.py`. Modify the mapping in `known_datasets` to point to your training and validation directory roots:

```python
known_datasets = {
    # ... other paths ...
    'hysp11k-ext-train': '/path/to/hyspecnet11k/data',
    'hysp11k-ext-val':   '/path/to/hyspecnet11k/test_data/test_data'
}
```

---

### 3. Pipeline Architecture Overview

The latent representations are sequentially modulated through our **Spectral-Spatial-Frequency Modulation Adapter (SSFMA)** at scales `enc_s8`, `enc_s16`, `enc_s32`, and `enc_s64`:

$$\mathbf{z} \xrightarrow{\text{SpMA}} \mathbf{z}_{\text{spec}} \xrightarrow{\text{SMA}} \mathbf{z}_{\text{spatial}} \xrightarrow{\text{FMA}} \mathbf{z}_{\text{task}}$$

* **Spectral Modulation Adapter (SpMA):** Squeeze-and-excitation channel attention to prioritize diagnostic spectral bands.
* **Spatial Modulation Adapter (SMA):** Depthwise-separable pixel attention mask to suppress terrain shadows, clouds, and non-semantic backgrounds.
* **Frequency Modulation Adapter (FMA):** Lightweight channel-wise **Spatial 2D-FFT** gating to isolate task-relevant frequency domains with minimal parameters.

#### Trainability Efficiency
* **Base HSI Codec:** Completely **frozen** during training.
* **Trainable Parameters:** **`1.0090%`** parameter overhead (only the sequential adapters and U-Net segmenter parameters are unfrozen).

---

### 4. Running the Training Pipeline (Co-Adaptation)

To initiate training on your RTX 4090 GPU server, execute:

```bash
python train_task_adapter.py
```

The script will:
1. Automatically detect and allocate CUDA execution on the RTX 4090 GPU.
2. Initialize `qarv_hsi_lower` in `adapted=True` mode (freezing the base codec and mounting active SSFMA modules).

## Pre-Training the Base Compression Model

If you do not have a pre-trained base VAE, you can train `qarv_hsi_lower` from scratch on the RTX 4090 GPU:

```bash
python train-var-rate.py \
    --model qarv_hsi_lower \
    --trainset hysp11k-ext-train \
    --valset hysp11k-ext-val \
    --iterations 500000 \
    --model_val_interval 5000 \
    --batch_size 16 \
    --workers 8 \
    --amp
```

- `--amp` ensures automatic mixed precision is used for speed.
- The base model must be trained first before you run the co-adaptation script.
4. Train under the joint RDT loss:
   $$L = \text{Rate} + \lambda_H \cdot \left[ (1 - \alpha) \cdot \text{MSE} + \alpha \cdot \text{SAM} \right] + \lambda_M \cdot L_{\text{task}}$$
5. Save the optimized parameter checkpoints to `adapters_and_task_head.pt`.

---

## 🛠️ File Structure

* **`lvae/models/qarv/adapters.py`:** Holds the SpMA, SMA, FMA, and SSFMA neural network layers.
* **`lvae/models/qarv/zoo_hsi.py`:** Registers the `adapted` loader mode and freezes base parameter weights.
* **`lvae/models/qarv/model.py`:** centralized forward pass integration (`forward_end2end`) that routes features through adapters.
* **`lvae/paths.py`:** Handles environment dataset path configurations.
* **`train_task_adapter.py`:** The training entrypoint, defining U-Net segmenter, stable numerical SAM loss, and data loading loops.
