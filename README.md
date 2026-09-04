# HVNet: Glaucoma Subtyping via Hypothesis-Guided Verification and Arbitrated Fusion

PyTorch implementation, experimental configurations, and inference pipeline accompanying **HVNet** (*Hypothesis-and-Verify Network*), a sequential multimodal framework for three-class glaucoma triage.

---

## 📖 Overview

Differentiating **Normal**, **open-angle glaucoma (OAG)**, and **angle-closure glaucoma (ACG)** is clinically important for subtype-oriented risk stratification and referral. However, definitive glaucoma subtype assessment generally relies on specialist examinations such as gonioscopy, ultrasound biomicroscopy (UBM), and anterior-segment optical coherence tomography (AS-OCT), which may not always be readily available in community or resource-constrained settings.

HVNet addresses this **anterior-posterior reasoning gap** by integrating a single posterior **color fundus photograph (CFP)** with three routinely available structured clinical indicators:

- **Intraocular pressure (IOP)**
- **Slit-lamp examination (SLE)-derived anterior chamber depth (ACD)**
- **Van Herick-based chamber-angle grade (CAG)**

Rather than combining the two modalities only through conventional parallel encoding and late fusion, HVNet follows a sequential **Hypothesize--Verify--Arbitrate** workflow:

1. **Hypothesize**  
   Encodes the structured clinical indicators into an initial triage prior using a lightweight Transformer encoder.

2. **Verify**  
   Uses the clinical hypothesis to guide multi-granularity visual feature extraction from CFP through the **Multi-Granularity Visual Attention (MGVA)** module.

3. **Arbitrate**  
   Integrates clinical and visual evidence using **Dirichlet evidential learning**, temperature-calibrated **Dempster--Shafer (DS) evidence fusion**, and a **sample-adaptive confidence gate**.

> **Terminology note:** In schematic figures and selected ablation settings, the label **Text** refers to the structured clinical branch formed by IOP, ACD, and CAG, rather than to free-text or natural-language input.

---

## 🧠 Architecture

<p align="center">
  <img src="assets/HVNet_architecture.png" width="95%">
</p>

The main components of HVNet are:

- **Clinical Hypothesis Encoder**
- **ConvNeXt-Tiny Visual Backbone**
- **Multi-Granularity Visual Attention (MGVA)**
  - Hypothesis-guided global cross-attention
  - Saliency-guided local region modeling
- **Dirichlet Evidential Representation**
- **Dempster--Shafer Evidence Fusion**
- **Sample-Adaptive Confidence Gating**
- **Joint Classification and Evidential Learning Objective**

The clinical branch first establishes a structured prior. This prior subsequently serves as a query for hypothesis-guided visual verification. Finally, the clinical, visual, and DS-fused predictions are adaptively weighted to generate the final three-class output.

---

## 🏆 Benchmark Performance

The internal cohort contained **2,281 eyes from 1,402 participants** and was divided at the patient level into an 80% development subset and a fixed 20% internal test subset.

Five-fold cross-validation was performed within the development subset. Each fold-specific model was subsequently evaluated on the same fixed internal test subset and on a held-out community-referral cohort comprising **279 eyes from 179 participants**.

| Evaluation Cohort | Accuracy (%) | Macro-Specificity (%) | Macro-AUC (%) | Cohen's Kappa (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Internal Test Set** | **92.06 ± 0.42** | **95.85 ± 0.25** | **98.04 ± 0.11** | **88.04 ± 0.63** |
| **Held-out Community-Referral Set** | **85.02 ± 2.83** | **92.51 ± 1.46** | **94.93 ± 0.42** | **77.49 ± 4.27** |

Results are reported as mean ± standard deviation across the five fold-specific models.

---

## 📁 Repository Structure

```text
HVNet/
├── README.md
├── requirements.txt
├── LICENSE
│
├── assets/
│   └── HVNet_architecture.png
│
├── configs/
│   └── default.yaml
│
├── models/
│   ├── hvnet.py
│   ├── clinical_encoder.py
│   ├── mgva.py
│   ├── ds_fusion.py
│   └── confidence_gate.py
│
├── datasets/
│   └── dataset_template.py
│
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── transforms.py
│
├── train.py
├── evaluate.py
└── inference.py
```

The exact repository structure may be updated as the implementation is further organized.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/HVNet.git
cd HVNet
```

### 2. Create a Python environment

For example:

```bash
conda create -n hvnet python=3.10
conda activate hvnet
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The experiments reported in the manuscript were implemented using **PyTorch 2.1** on **Ubuntu 20.04 LTS** and trained on a single **NVIDIA GeForce RTX 4090 GPU**.

---

## 📥 Input Format

Each HVNet sample consists of:

1. One CFP image;
2. Three structured clinical indicators:
   - **IOP status:** elevated (>21 mmHg) or not;
   - **SLE-derived ACD:** normal or shallow;
   - **Van Herick CAG:** grade 0--4;
3. One reference-standard class label:
   - Normal;
   - OAG;
   - ACG.

A generic tabular representation may follow the format:

```text
image_path,iop,acd,cag,label
/path/to/example.jpg,1,0,3,target_label
```

The exact numerical class encoding is defined in the dataset loader and configuration files.

No patient identifiers or protected health information should be included in input files used with the public implementation.

---

## 🚀 Quick Start

### Training

```bash
python train.py --config configs/default.yaml
```

### Evaluation

```bash
python evaluate.py \
    --config configs/default.yaml \
    --checkpoint path/to/checkpoint.pth
```

### Inference

```bash
python inference.py \
    --image path/to/fundus.jpg \
    --iop 1 \
    --acd 0 \
    --cag 3 \
    --checkpoint path/to/checkpoint.pth
```

Command-line arguments may be updated according to the final implementation.

---

## ⚙️ Hyperparameter Configuration

The main experimental settings are centralized in:

```text
configs/default.yaml
```

The configuration corresponding to the manuscript includes:

### Visual Backbone

- **Backbone:** ConvNeXt-Tiny
- **Pretraining:** ImageNet-1K
- **Input resolution:** $224 \times 224$ pixels

### Optimization

- **Optimizer:** AdamW
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- **Weight decay:** 0.01
- **Batch size:** 32
- **Training epochs:** 80
- **Gradient clipping:** maximum norm of 5.0

### Learning Rates

- **Visual backbone:** $5 \times 10^{-5}$
- **Remaining modules:** $5 \times 10^{-4}$
- **Scheduler:** cosine annealing
- $\eta_{\min} = 5 \times 10^{-7}$

### Two-Stage Optimization

- Epochs 1--5: visual backbone frozen
- Epochs 6--80: joint end-to-end optimization

### MGVA and Evidential Fusion

- **Number of selected local regions:** $\mathcal{K}=4$
- **DS evidence temperature:** $\tau=1.0$
- **Auxiliary classification weight:** $\lambda_{\mathrm{aux}}=0.3$
- **Evidential regularization weight:** $\lambda_{\mathrm{edl}}=0.1$
- **Confidence-gate output bias initialization:** `[1.0, -0.5, -0.5]`

### Reproducibility

- **Random seed:** `42`
- Data partitioning is performed at the **patient level** to prevent leakage between development and test subsets.

---

## 🧪 Training Protocol

The internal cohort is divided at the patient level into:

- **80% development subset**
- **20% fixed internal test subset**

Five-fold cross-validation is performed within the development subset.

For each fold:

1. Four folds are used for training;
2. One fold is used for validation and checkpoint selection;
3. The checkpoint with the highest validation accuracy is retained;
4. The selected model is evaluated on:
   - the fixed internal test subset;
   - the held-out community-referral cohort.

Neither test cohort is used for model configuration, hyperparameter selection, or checkpoint selection.

---

## 📊 Evaluation Metrics

The following metrics are reported:

- **Accuracy (Acc)**
- **Macro-specificity (Spec)**
- **Macro one-vs-rest AUC**
- **Cohen's kappa**

Performance is summarized as the mean ± standard deviation across the five fold-specific models.

---

## 🛡️ Data and Code Availability

### Source Code

This repository provides the core implementation of HVNet, including the major network components, experimental configuration, evidential learning objective, training pipeline, and evaluation procedures.

The repository does **not** contain patient-level clinical data or identifiable health information.

### Clinical Data

The clinical datasets generated and/or analyzed in this study are not publicly available because of patient privacy, ethical considerations, and institutional data-governance requirements.

Access may be considered upon reasonable academic request and is subject to approval by the relevant ethics committee and participating institutions.

The retrospective study was approved by the Ethics Committee of **The Affiliated People's Hospital of Ningbo University, China**  
(Approval No. **2024-067**).

Approved access may additionally be subject to institutional data-use requirements.

### Model Weights

Pretrained model weights are not currently distributed in this repository.

Their future release, if applicable, will be subject to institutional and collaborative approval.

---

## 🔬 Reproducibility Notes

Because the original clinical dataset cannot be publicly distributed, complete numerical reproduction of the manuscript results requires access to the corresponding study data.

The public implementation is intended to provide:

- the complete HVNet architecture;
- the clinical encoding strategy;
- the MGVA module;
- the evidential learning formulation;
- DS evidence fusion;
- confidence gating;
- training configurations;
- evaluation procedures;
- dataset-interface templates.

Users may adapt the provided data loader to their own CFP and structured clinical datasets.

---

## ⚠️ Intended Use

HVNet is provided for **research purposes only**.

The framework was developed for subtype-oriented glaucoma triage and is intended to support early risk stratification and referral research.

It is **not** intended to:

- replace gonioscopy or specialist glaucoma assessment;
- provide autonomous clinical diagnosis;
- determine treatment decisions;
- replace established glaucoma diagnostic workflows.

The current results should not be interpreted as evidence of clinical effectiveness or readiness for autonomous deployment.

Independent external validation and prospective clinical evaluation are required before clinical use.

---

## 📝 Citation

This repository accompanies the manuscript:

> **Glaucoma Subtyping via Hypothesis-Guided Verification and Arbitrated Fusion**

Citation information will be updated upon publication.

A BibTeX entry will be provided after the article receives its final bibliographic information.

---

## 📬 Contact

For questions regarding the implementation, please open an issue in this repository.

For questions regarding the study, clinical data access, or institutional collaboration, please contact the corresponding authors listed in the manuscript.

---

## 📄 License

Licensing terms will be specified following institutional and collaborative approval.

Please do not redistribute clinical data, model weights, or other restricted study materials without appropriate authorization.

---

## 🙏 Acknowledgements

We thank the participating clinical institutions, ophthalmologists, and research collaborators involved in data collection, reference-standard adjudication, and model development.

---

## 📌 Repository Status

This repository is being prepared to accompany the HVNet manuscript.

Core implementation files, configuration files, and documentation will be updated as the code is finalized for public release.
