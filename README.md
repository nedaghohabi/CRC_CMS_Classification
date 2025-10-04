# Image-Based CMS Classification of Colorectal Cancer (ResNet-34)

This repository contains code and configs for our study:
**“Image-Based Consensus Molecular Subtype Classification of Colorectal Cancer using Deep Learning.”**

We train a ResNet-34 CNN on tumor tiles from TCGA COAD/READ histopathology whole-slide images (WSIs) to predict **CMS1–CMS4** subtypes. Labels are derived from RNA-seq using the **CMSclassifier** package. The repo includes:
- Training & evaluation on COAD/READ (10-fold CV + held-out test)
- Tile-level inference + **sample-level majority voting** for **unclassified** cases
- Pointers to  **WSI preprocessing** and **transcriptomic CMS label generation**

> **Note**: This code is research-grade. External validation is required before any clinical use.

---

## Contents
- [Overview](#overview)
- [Data & Preprocessing](#data--preprocessing)
- [CMS Label Generation (RNA-seq)](#cms-label-generation-rna-seq)
- [Environment](#environment)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Training & Evaluation](#training--evaluation)
- [Prediction for Unclassified Samples](#prediction-for-unclassified-samples)
- [Results Artifacts](#results-artifacts)
- [Citations](#citations)
- [License](#license)

---

## Overview

- **Backbone**: ResNet-34 (ImageNet-pretrained), first conv optionally replaced with 5×5 (stride=2, pad=2)
- **Input**: tiles **224×224** randomly cropped from **512×512** tumor patches
- **Cohorts**: TCGA **COAD** and **READ**, trained **separately** with identical settings
- **Split**: Default **10-fold CV** on tiles + **10% held-out test**
- **Metrics**: Accuracy, Precision/Recall/F1, **ROC-AUC** (per-class, micro, macro), Confusion Matrix
- **Post-hoc**: Predict **unclassified** samples via tile-level thresholds + majority voting

---

## Data & Preprocessing

WSIs and RNA-seq come from **TCGA**.

- **WSI Source**: TCGA (GDC)  
  - Portal: https://portal.gdc.cancer.gov/
  - Download format: **.svs**
- **Tumor ROI annotation & tiling**: We used QuPath scripting to annotate tumor regions (pathologist-supervised), tessellate into **512×512** tiles at 20×, remove non-informative tiles, and export JPGs.

**Preprocessing helper repo** (annotation + tiling scripts):  
**[ADD_LINK_HERE to the external preprocessing repository]**

Expected folder layout after preprocessing (example):


---

## CMS Label Generation (RNA-seq)

Labels are derived using **CMSclassifier** on TCGA RNA-seq (FPKM-UQ), following the **RF + SSP consensus** approach.

**CMS label generation helper repo/script**:  
**[ADD_LINK_HERE to the external CMSclassifier R repository/script]**

We used:
- **TCGAbiolinks** to retrieve RNA-seq & clinical data
- **edgeR / limma** default normalization on FPKM-UQ
- **CMSclassifier** (RF posterior ≥ 0.5) and **SSP** after row-centering
- Final label retained only if **RF == SSP**; otherwise **unclassified**

---

## Environment

- **Python**: 3.9+
- **PyTorch**: 2.x (CUDA 11.8 recommended)
- **R** (CMSclassifier; run in the external repo)
- **QuPath** 0.3.2 (annotations/tiling)

Create a conda env (example):
```bash
conda create -n cms-imaging python=3.9 -y
conda activate cms-imaging
pip install -r requirements.txt

.
├── configs/
│   ├── coad.yaml
│   └── read.yaml
├── data/                      # (not tracked) tiles organized by CMS class
├── labels/
│   ├── coad_labels.csv
│   └── read_labels.csv
├── src/
│   ├── datasets.py            # Tile dataset, transforms, samplers
│   ├── model.py               # ResNet-34 builder (optionally 5×5 first conv)
│   ├── train_eval.py          # k-fold CV + held-out test
│   ├── predict_unclassified.py# tile-level inference + majority voting
│   ├── metrics.py             # AUC, reports, confusions
│   └── utils.py               # logging, seed, checkpoint IO
├── results/
│   ├── coad/
│   │   ├── cv/                # per-fold metrics, ROC, confusion
│   │   └── test/              # held-out test metrics & plots
│   └── read/
│       ├── cv/
│       └── test/
├── assets/
│   └── figures/
│       └── workflow_main_1.png
├── requirements.txt
├── README.md
└── LICENSE


requirements.txt


torch
torchvision
torchaudio
numpy
pandas
scikit-learn
scipy
matplotlib
tqdm
pyyaml
opencv-python

image_size: 224
tile_size: 512
batch_size: 16
epochs: 5

model:
  backbone: resnet34
  imagenet_pretrained: true
  first_conv_5x5: true
  num_classes: 4

optimizer:
  name: adam
  lr: 1e-4
  weight_decay: 5e-4
  amsgrad: true

cv:
  folds: 10
  split_level: tile 
  test_fraction: 0.10
  stratified: true

augment:
  hflip_prob: 0.5
  vflip_prob: 0.5
  rotate_deg: 45

inference:
  prob_threshold: 0.5
  voting: majority



Outputs:
Per-fold metrics: accuracy, precision/recall/F1, ROC-AUC (per-class, micro, macro)
Plots: ROC curves & confusion matrices (saved under results/<cohort>/cv/)
Best fold checkpoint → evaluated on held-out test set (saved under results/<cohort>/test/)
