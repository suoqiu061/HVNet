# HVNet: Glaucoma Subtyping via Hypothesis-Guided Verification and Arbitrated Fusion

Official PyTorch implementation, experimental configurations, and inference pipeline for **HVNet** (*Hypothesis-and-Verify Network*).

---

## 📖 Overview

Glaucoma subtype differentiation (Normal vs. Open-Angle Glaucoma [**OAG**] vs. Angle-Closure Glaucoma [**ACG**]) is critical for subtype-tailored clinical referrals and management. However, definitive subtype assessment relies on specialist procedures (e.g., gonioscopy, UBM, AS-OCT) that are scarce in primary care, while conventional color fundus photography (CFP) AI models are restricted to binary glaucoma detection.

To address this **anterior-posterior reasoning gap**, HVNet couples a single posterior CFP with three routinely accessible structured clinical indicators (IOP, SLE-derived ACD, and Van Herick CAG) through a three-stage sequential reasoning pipeline:

1. **Hypothesize Stage:** Maps the anterior clinical indicators into an initial diagnostic hypothesis prior via a lightweight Transformer encoder.
2. **Verify Stage:** Conditions posterior visual feature extraction on the clinical hypothesis using a **Multi-Granularity Visual Attention (MGVA)** module, balancing global semantic scales and local salient retinal regions.
3. **Arbitrate Stage:** Synthesizes multimodal evidence through **Dirichlet Evidential Deep Learning (EDL)**, temperature-calibrated **Dempster–Shafer (DS) evidence fusion**, and dynamic **sample-adaptive confidence gating**.

---

## 🏆 Benchmark Performance

Evaluated under strict patient-level 5-fold cross-validation on an internal development cohort ($n=1,825$ eyes), a fixed internal test cohort ($n=456$ eyes), and a held-out multicenter community-referral cohort ($n=279$ eyes):

| Evaluation Cohort | Accuracy (%) | Macro-Specificity (%) | Macro-AUC (%) | Cohen's Kappa (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Internal Test Set** | **92.06 ± 0.42** | **95.85 ± 0.25** | **98.04 ± 0.11** | **88.04 ± 0.63** |
| **Held-out Referral Set** | **85.02 ± 2.83** | **92.51 ± 1.46** | **94.93 ± 0.42** | **77.49 ± 4.27** |

---

## 🛡️ Data and Code Availability Statement

To ensure scientific transparency and community reproducibility:
- **Source Code:** Core network architectures (`models/`), evidential objective losses (`utils/losses.py`), and validation pipelines are fully disclosed in this repository.
- **Clinical Datasets & Weights:** The de-identified multicenter clinical CFP images and tabular electronic records are available from the corresponding authors upon reasonable academic request, subject to institutional ethical review (Ethics Committee of The Affiliated People's Hospital of Ningbo University, Approval No. 2024-067) and execution of a formal Data Use Agreement (DUA).

---

## ⚙️ Hyperparameter Configurations & Reproducibility

All experimental configurations are centralized in `configs/default.yaml`:

- **Visual Backbone:** ConvNeXt-Tiny (Pretrained on ImageNet-1K)
- **Image Input Size:** $224 \times 224$ pixels
- **Optimization:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 0.01$)
- **Learning Rate Schedule:** Cosine Annealing ($\eta_{\min}=5\times 10^{-7}$)
  - Backbone LR: $5 \times 10^{-5}$
  - Head & Transformer LR: $5 \times 10^{-4}$
- **Two-Phase Optimization:** Backbone frozen for the first 5 epochs; unfrozen for joint end-to-end training across Epochs 6–80.
- **Batch Size:** 32 (Trained on a single NVIDIA RTX 4090 GPU)
- **Evidential & Fusion Hyperparameters:**
  - DS Evidence Calibration Temperature ($\tau$): $1.0$
  - Number of Local Salient Patches ($\mathcal{K}$): $4$
  - Auxiliary Loss Weight ($\lambda_{\mathrm{aux}}$): $0.3$
  - Dirichlet Evidential KL Weight ($\lambda_{\mathrm{edl}}$): $0.1$
  - Gate MLP Initial Bias: `[1.0, -0.5, -0.5]` (mild clinical prior bias)
- **Deterministic Repro:** Global random seed fixed to `42`.

---
