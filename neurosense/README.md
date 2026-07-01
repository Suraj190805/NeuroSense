<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/MONAI-1.3-00B2A9?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgo=&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-19.x-61DAFB?style=for-the-badge&logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
</p>

# 🧬 NeuroSense

**AI-Powered Huntington's Disease Detection & Progression Analysis Platform**

NeuroSense is a multi-modal clinical decision support framework that integrates
structural MRI, genetic biomarkers (CAG repeat count), and UHDRS clinical scores
to enable early Huntington's Disease detection and 12/24-month progression forecasting.

> ⚠️ **Disclaimer**: This is an academic research project. AI predictions are not
> a substitute for professional medical diagnosis. All datasets are de-identified
> per HIPAA Safe Harbor standards.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **3D ResNet-50 MRI Encoder** | Volumetric brain feature extraction using MONAI with Med3D transfer learning |
| 📈 **Bi-LSTM Clinical Encoder** | Longitudinal visit sequence modelling for temporal clinical patterns |
| 🔀 **Cross-Modal Attention Fusion** | Transformer-based fusion where imaging features are weighted by clinical context |
| 🎯 **Three-Class HD Staging** | Pre-manifest / Early / Advanced classification with calibrated confidence |
| 📊 **Progression Forecasting** | 12- and 24-month UHDRS motor score trajectory prediction |
| 🔥 **GradCAM++ Heatmaps** | 3D spatial activation maps overlaid on axial MRI slices |
| 📋 **SHAP Feature Attribution** | Per-feature importance for CAG repeat, UHDRS subscores, TFC |
| 🌐 **FastAPI Backend** | Async prediction endpoint with Pydantic validation |
| 💎 **React Clinical Dashboard** | Premium dark-mode interface with glassmorphism design |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      NeuroSense Pipeline                        │
├──────────┬──────────────────┬───────────────────────────────────┤
│          │                  │                                   │
│  ┌───────▼────────┐  ┌─────▼──────────┐                       │
│  │  MRI Input     │  │ Clinical Input  │                       │
│  │  (.nii.gz)     │  │ (CAG, UHDRS)   │                       │
│  └───────┬────────┘  └─────┬──────────┘                       │
│          │                  │                                   │
│  ┌───────▼────────┐  ┌─────▼──────────┐                       │
│  │ 3D ResNet-50   │  │  Bi-LSTM       │                       │
│  │ MRI Encoder    │  │  Clinical Enc. │                       │
│  │ → 256-dim emb  │  │  → 256-dim emb │                       │
│  └───────┬────────┘  └─────┬──────────┘                       │
│          │                  │                                   │
│          └────────┬─────────┘                                   │
│                   │                                             │
│          ┌────────▼─────────┐                                   │
│          │  Cross-Modal     │                                   │
│          │  Attention Fusion│                                   │
│          │  (8-head MHA)    │                                   │
│          └────────┬─────────┘                                   │
│                   │                                             │
│          ┌────────▼─────────┐                                   │
│          │  256-dim Fused   │                                   │
│          │  Representation  │                                   │
│          └───┬──────────┬───┘                                   │
│              │          │                                       │
│     ┌────────▼───┐  ┌───▼──────────┐                           │
│     │ Stage      │  │ Progression  │                           │
│     │ Classifier │  │ Forecaster   │                           │
│     │ (3-class)  │  │ (12/24 mo)   │                           │
│     └────────┬───┘  └───┬──────────┘                           │
│              │          │                                       │
│     ┌────────▼───┐  ┌───▼──────────┐                           │
│     │ GradCAM++  │  │ SHAP         │                           │
│     │ Heatmaps   │  │ Waterfall    │                           │
│     └────────────┘  └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
neurosense/
├── data/                    # Dataset classes, preprocessing, BIDS loaders
│   ├── dataset.py           # HuntingtonDataset (MONAI-based)
│   └── preprocessing.py     # MRI + tabular preprocessing pipeline
├── models/                  # Neural network architectures
│   ├── mri_encoder.py       # 3D ResNet-50 with projection head
│   ├── clinical_encoder.py  # Bi-LSTM for longitudinal sequences
│   ├── fusion.py            # Cross-modal attention module
│   └── classifier.py        # Classification + progression heads
├── training/                # Training & evaluation
│   ├── train.py             # End-to-end training loop
│   ├── ablation.py          # 5-condition ablation study
│   ├── evaluate.py          # Metrics: AUC, F1, ECE
│   └── losses.py            # Weighted CE + Huber loss
├── explainability/          # XAI modules
│   ├── gradcam.py           # GradCAM++ for 3D MRI volumes
│   ├── shap_analysis.py     # SHAP DeepExplainer
│   └── visualise.py         # Heatmap overlay generation
├── api/                     # FastAPI backend
│   ├── main.py              # POST /predict, GET /health
│   ├── schemas.py           # Pydantic request/response models
│   └── inference.py         # Inference pipeline orchestration
├── frontend/                # React clinical dashboard
├── configs/                 # YAML configuration files
│   ├── model_config.yaml    # Architecture hyperparameters
│   └── train_config.yaml    # Training hyperparameters
├── scripts/                 # Setup & utility scripts
│   ├── setup_env.sh         # One-command environment setup
│   ├── download_data.sh     # Dataset download helper
│   └── preprocess_all.sh    # MRI preprocessing pipeline
├── tests/                   # Unit & integration tests
├── notebooks/               # EDA & ablation result notebooks
├── requirements.txt         # Python dependencies (pinned)
├── environment.yml          # Conda environment spec
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version | Required |
|---|---|---|
| **Python** | 3.10+ | ✅ |
| **CUDA** | 11.8+ | Recommended (GPU training) |
| **Node.js** | 18+ | For frontend dashboard |
| **FSL** | 6.0+ | Optional (skull stripping) |

### 1. Clone & Setup

```bash
git clone https://github.com/Suraj190805/NeuroSense.git
cd neurosense

# One-command setup: creates venv, installs all dependencies
bash scripts/setup_env.sh

# Activate environment
source venv/bin/activate
```

### 2. Download Data

```bash
# Option A: OpenNeuro ds004040 (public, for development)
bash scripts/download_data.sh --openneuro

# Option B: PREDICT-HD (requires FITBIR approval — allow 2–4 weeks)
bash scripts/download_data.sh --predict-hd
```

### 3. Preprocess MRI Data

```bash
bash scripts/preprocess_all.sh
```

### 4. Train the Model

```bash
# Full multi-modal training
python -m neurosense.training.train

# Run ablation study (5 conditions × 3 seeds)
python -m neurosense.training.ablation
```

### 5. Start the API

```bash
uvicorn neurosense.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Launch the Dashboard

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## 📡 API Reference

### `POST /predict`

Upload MRI + clinical data for HD staging and progression prediction.

**Request** (`multipart/form-data`):

| Field | Type | Required | Range |
|---|---|---|---|
| `mri_file` | File | ✅ | .nii or .nii.gz |
| `cag_repeat` | float | ✅ | 36–120 |
| `uhdrs_motor` | float | ✅ | 0–124 |
| `uhdrs_cognitive` | float | ✅ | ≥ 0 |
| `tfc_score` | float | ❌ | 0–13 |
| `age` | float | ✅ | 18–90 |

**Response** (`application/json`):

```json
{
  "stage": "early",
  "confidence": 0.87,
  "stage_probabilities": {
    "pre_manifest": 0.08,
    "early": 0.87,
    "advanced": 0.05
  },
  "progression_12mo": 4.2,
  "progression_24mo": 9.1,
  "risk_category": "medium",
  "gradcam_url": "/static/heatmaps/abc123.png",
  "shap_features": [
    {"name": "cag_repeat", "value": 44.0, "impact": 0.32},
    {"name": "uhdrs_motor", "value": 18.0, "impact": 0.28},
    {"name": "tfc", "value": 9.0, "impact": -0.15},
    {"name": "uhdrs_cognitive", "value": 142.0, "impact": 0.12},
    {"name": "age", "value": 42.0, "impact": 0.08}
  ],
  "processing_time_s": 12.4
}
```

### `GET /health`

Returns service health status and GPU availability.

### `GET /version`

Returns API version and model checkpoint hash.

---

## 🧪 Ablation Study

| Condition | MRI | Clinical | Expected AUC |
|---|---|---|---|
| MRI only (baseline) | ✓ | ✗ | ~0.78–0.82 |
| Clinical only | ✗ | ✓ | ~0.74–0.78 |
| Genetic only (CAG) | ✗ | CAG only | ~0.70–0.75 |
| MRI + Clinical (concat) | ✓ | ✓ | ~0.82–0.85 |
| **Full NeuroSense (fusion)** | ✓ | ✓ | **≥ 0.87** |

Each condition runs with 3 random seeds. Results logged to `ablation_results.csv`.

---

## 🔧 Technology Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.x + MONAI 1.3 |
| Neuroimaging | FSL, ANTs, NiBabel |
| Explainability | pytorch-grad-cam, SHAP |
| Backend | FastAPI + Uvicorn |
| Frontend | React 19 + Recharts + Framer Motion |
| Experiment Tracking | Weights & Biases |
| Environment | Python 3.10 + CUDA 11.8 |

---

## 📄 License

This project is for academic research purposes. All datasets are used under
their respective data use agreements.

---

## 👤 Author

**Suraj S** 

---

<p align="center">
  <strong>Built with 🧠 AI + 🧬 Neuroscience for early Huntington's Disease detection</strong>
</p>
