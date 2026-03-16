# 🦴 Lumbar Spine Degeneration Classification using Deep Learning

> An end-to-end deep learning pipeline for automated classification of lumbar spine degeneration severity from MRI scans — classifying conditions as Normal/Mild, Moderate, or Severe.
---

## 📌 Project Overview

| Detail | Info |
|---|---|
| **Dataset** | RSNA 2024 Lumbar Spine Degenerative Classification (Kaggle) |
| **MRI Studies** | ~1,975 training studies, 18,000+ labeled spinal levels |
| **Task** | 3-class severity classification (Normal/Mild, Moderate, Severe) |
| **Best Model** | EfficientNet-B3 — 65% accuracy, Weighted F1: 0.65 |
| **Language** | Python (PyTorch) |

---

## 🧠 Problem Statement

Lumbar spine degeneration is a leading cause of chronic back pain affecting millions globally. Radiologists manually review hundreds of MRI slices — a slow, subjective, and fatigue-prone process.

This project develops an **AI-powered classification pipeline** to:
- Classify 3 critical conditions: Spinal Canal Stenosis, Neural Foraminal Narrowing, Subarticular Stenosis
- Assess severity across 5 vertebral levels: L1/L2 through L5/S1
- Assist radiologists with faster, more consistent diagnoses

---

## 🔬 Preprocessing Pipeline
```
DICOM Loading → Sagittal T2 Slice Selection → CLAHE Enhancement
→ Normalization → Resize (256×256) → 2.5D Tensor Stacking
→ Augmentation → Class Weight Computation → Model Input
```

**Key techniques:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement
- 2.5D representation: stacking 3 middle sagittal slices into a tensor
- Class weights to handle severe imbalance (Normal/Mild : Severe = 12.4 : 1)

---

## 🤖 Models Compared

| Model | Train Acc | Val Acc | Weighted F1 | Severe Recall |
|---|---|---|---|---|
| Baseline CNN | 52.78% | 52% | 0.56 | 0.10 |
| ResNet50 | 81.39% | 62% | 0.64 | 0.29 |
| **EfficientNet-B3** ⭐ | **82.34%** | **65%** | **0.65** | **0.31** |
| Vision Transformer (ViT) | 82.72% | 62% | 0.64 | 0.29 |
| Hybrid CNN-Transformer | 84% | 61% | 0.63 | 0.40 |

---

## 📊 Best Model: EfficientNet-B3
```
              precision  recall  f1-score  support
Normal/Mild      0.80     0.79     0.80      292
Moderate         0.20     0.20     0.20       45
Severe           0.29     0.31     0.30       58
Accuracy                           0.65      395
Weighted avg     0.66     0.65     0.65      395
```

---

## 💡 Key Findings

- **Transfer learning is essential** — pretrained models outperformed baseline by 10–13%
- **Model size must match data scale** — EfficientNet-B3 (12M params) outperformed ViT (86M params)
- **Class imbalance remains challenging** — severe recall only 31% despite weighted loss
- **CNNs beat Transformers** on this limited medical dataset

---

## 🗃️ Dataset

| Detail | Info |
|---|---|
| **Source** | RSNA 2024 via Kaggle |
| **Format** | DICOM MRI files |
| **Sequences** | Sagittal T1, Sagittal T2/STIR, Axial T2 |
| **Labels** | 25 condition-level combinations × 3 severity grades |
| **Class Ratio** | Normal/Mild : Severe ≈ 12.4 : 1 |

> ⚠️ Dataset is not included in this repo due to size. Download from [Kaggle RSNA 2024](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)

---

## ⚙️ How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download RSNA 2024 dataset from Kaggle and place in `/data` folder

3. Run the notebook:
```bash
jupyter notebook Copy_of_Deep_Learning_Project.ipynb
```

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![EfficientNet](https://img.shields.io/badge/EfficientNet-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## 🔮 Future Work

- Train on complete dataset (currently used 40% subset)
- Incorporate Axial T2 + Sagittal T1 sequences
- Implement 3D CNNs for volumetric analysis
- Add Grad-CAM interpretability for radiologist trust
- Advanced class imbalance techniques (focal loss, synthetic augmentation)
- Multi-label learning for all 25 conditions

---

## 📁 Repository Structure
```
lumbar-spine-degeneration-classification/
├── Copy_of_Deep_Learning_Project.ipynb   # Main notebook
├── Final_Project_Report.pdf              # Full technical report
├── Lumbar-Spine-Degeneration-Classification.pdf  # Presentation slides
├── requirements.txt                      # Dependencies
└── README.md                             # This file
```

---

## 👤 Author

**Mayank Choudhary**
MS Data Science | Stevens Institute of Technology

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mayankchoudhary23/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:mayankchoudharystevens909@gmail.com)
```
