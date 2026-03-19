# ESED: An Explainable Stacked Ensemble Framework for Multi-Class Pulmonary Disease Diagnosis from Chest X-Rays

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://kaggle.com)

## Overview

ESED (Explainable Stacked Ensemble for Disease Diagnosis) is a 
novel deep learning framework for automatic multi-class pulmonary 
disease diagnosis from chest X-rays. The framework combines five 
architecturally diverse convolutional neural networks as base 
learners with an XGBoost meta-learner, augmented by comprehensive 
explainability analysis through Grad-CAM, SHAP, and LIME.

**Status:** Under review at a peer-reviewed Q1 journal (2026)

---

## Key Results

| Model | Accuracy | F1-Macro | AUC-ROC |
|---|---|---|---|
| DenseNet201 | 0.9717 | 0.9716 | 0.9987 |
| EfficientNetB4 | 0.9537 | 0.9538 | 0.9968 |
| ResNet50V2 | 0.9696 | 0.9696 | 0.9978 |
| InceptionV3 | 0.9496 | 0.9497 | 0.9962 |
| ConvNeXtTiny | 0.9679 | 0.9679 | 0.9977 |
| **ESED Ensemble** | **0.9838** | **0.9838** | **0.9993** |

### Per-Class F1 (ESED Ensemble)
| COVID-19 | Pneumonia | TB | Normal |
|---|---|---|---|
| 0.98 | 0.99 | 0.99 | 0.97 |

### External Validation
| Dataset | Institution | Country | F1 | AUC |
|---|---|---|---|---|
| Internal test | Multi-source | Multi | 0.9892 | 0.9993 |
| Epic Chittagong | Epic Hospital | Bangladesh | 0.8381 | 0.9600 |
| NIH ChestX-ray14 | NIH Clinical Center | USA | 0.0222 | 0.4707 |

---

## Framework Architecture
```
Input X-Ray Image
       │
       ▼
┌─────────────────────────────────────────┐
│         Five Base Learners              │
│  DenseNet201 │ EfficientNetB4           │
│  ResNet50V2  │ InceptionV3              │
│              │ ConvNeXtTiny             │
│   (Two-phase transfer learning)         │
└─────────────────────────────────────────┘
       │
       ▼ Level-1 features (N×20)
┌─────────────────────────────────────────┐
│      XGBoost Meta-Learner               │
│   CV F1: 0.9936 ± 0.0009               │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│      XAI Explainability                 │
│   Grad-CAM │ SHAP │ LIME               │
└─────────────────────────────────────────┘
       │
       ▼
  Final Diagnosis + Confidence Score
  + Clinical Referral Flag (if conf < 0.95)
```

---

## Dataset

The ESED framework was trained on a curated multi-source dataset 
of 15,898 unique chest X-ray images (after MD5 deduplication) 
across four classes:

| Class | Images | Sources |
|---|---|---|
| COVID-19 | 3,949 | COVID-19 Radiography Database |
| Pneumonia | 3,976 | Paul Mooney Chest X-Ray |
| Tuberculosis | 3,973 | TB DS4 + TBX11K + Mendeley TB |
| Normal | 4,000 | Multiple sources |

**Dataset available at:**  
https://kaggle.com/datasets/mdkawshermahbub/pulmonary-merged-data

**Pre-trained models available at:**  
https://kaggle.com/datasets/mdkawshermahbub/pulmonary-model-outputs

---

## Requirements
```bash
pip install -r requirements.txt
```
```
tensorflow==2.19.0
keras
numpy
pandas
scikit-learn
xgboost
shap
lime
matplotlib
seaborn
opencv-python
scipy
joblib
```

---

## Reproduction

### Option 1 — Kaggle Notebook (Recommended)
1. Go to: https://www.kaggle.com/code/mdkawshermahbub/esed-pulmonary-xai
2. Fork the notebook
3. Add datasets:
   - `mdkawshermahbub/pulmonary-merged-data`
   - `mdkawshermahbub/pulmonary-splits`
   - `mdkawshermahbub/pulmonary-model-outputs`
4. Run all cells in order

### Option 2 — Local Reproduction
```bash
git clone https://github.com/[username]/ESED-Pulmonary-Disease-Diagnosis
cd ESED-Pulmonary-Disease-Diagnosis
pip install -r requirements.txt
# Download datasets from Kaggle
# Update paths in config
python src/base_learner_training.py
python src/ensemble_learning.py
python src/statistical_validation.py
python src/xai_analysis.py
```

---

## Statistical Validation

| Test | Result | Conclusion |
|---|---|---|
| McNemar | All 5 p<0.05 | Ensemble beats all base models |
| Friedman | χ²=29.12, p=0.002 | Meta-learner choice significant |
| Nemenyi | XGBoost vs MLP p=0.031 | XGBoost selection justified |
| Bootstrap CI | [0.9787, 0.9887] | Non-overlapping with all base CIs |

---

## XAI Summary

| Method | Model | Key Finding |
|---|---|---|
| Grad-CAM | All 5 models | Lower lung fields dominant |
| SHAP | DenseNet201 | 54–66% attribution in lower fields |
| LIME | DenseNet201 | COVID peak weight 0.4402 |

---

## Clinical Referral Strategy

At confidence threshold 0.95:
- Cases referred to radiologist: **2.9%**
- Accuracy on retained cases: **0.9923**
- Errors avoided: **21 of 39 (54%)**

---

## Data Integrity

- MD5 hash deduplication: 102 duplicates removed
- Cross-split leakage: 0 after deduplication
- Sensitivity analysis: F1 difference = 0.0002 (negligible)

---

## Citation

If you use this work please cite:
```bibtex
@article{mahbub2026esed,
  title={ESED: An Explainable Stacked Ensemble Framework 
         for Multi-Class Pulmonary Disease Diagnosis 
         from Chest X-Rays},
  author={Mahbub, MK and others},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

**MK Mahbub**  
PhD Student, University of Alabama  
Email: [your email]  
Kaggle: https://kaggle.com/mdkawshermahbub
```

---

### requirements.txt
```
tensorflow==2.19.0
keras>=3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.44.0
lime>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
scipy>=1.11.0
joblib>=1.3.0
Pillow>=10.0.0
