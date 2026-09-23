# 🧬 NeuroSense

<p align="center">
  <strong>AI-Powered Multi-Modal Huntington's Disease Staging & Progression Analysis Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x" />
  <img src="https://img.shields.io/badge/MONAI-1.3-00B2A9?style=for-the-badge&logo=monai&logoColor=white" alt="MONAI 1.3" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19.x-61DAFB?style=for-the-badge&logo=react&logoColor=white" alt="React 19" />
  <img src="https://img.shields.io/badge/Three.js-3D%20Viewer-black?style=for-the-badge&logo=threedotjs&logoColor=white" alt="Three.js" />
  <img src="https://img.shields.io/badge/MongoDB-Motor-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB" />
  <img src="https://img.shields.io/badge/Groq-AI%20Chatbot-F55036?style=for-the-badge&logo=openai&logoColor=white" alt="Groq AI" />
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA 11.8" />
</p>

---

> ⚠️ **Clinical Research Disclaimer**: NeuroSense is developed as an academic and clinical research platform. Model predictions, digital test scores, and progression trajectories are intended for decision-support and investigational purposes only and do not replace professional neurological evaluation or medical diagnosis. All neuroimaging and clinical records are handled in adherence with HIPAA Safe Harbor de-identification standards.

---

## 📌 Overview

**NeuroSense** is a comprehensive clinical decision support ecosystem engineered for the early detection, multi-class staging, and longitudinal progression forecasting of **Huntington’s Disease (HD)**. 

Huntington's Disease is an autosomal-dominant, progressive neurodegenerative disorder caused by an expanded CAG trinucleotide repeat in the *HTT* gene. Its clinical manifestation spans progressive motor dysfunction (chorea, bradykinesia, dystonia), cognitive decline (executive dysfunction, impaired processing speed), and psychiatric symptoms.

NeuroSense addresses the challenge of heterogeneous disease progression by synergizing:
1. **Volumetric Neuroimaging**: 3D structural T1-weighted brain MRI scans capturing striatal, caudate, and cortical atrophy.
2. **Genomic Biomarkers**: CAG repeat length and demographic risk variables.
3. **Longitudinal Clinical Metrics**: Unified Huntington's Disease Rating Scale (UHDRS) motor/cognitive scores and Total Functional Capacity (TFC).
4. **Digital Assessment Batteries**: In-browser neuropsychological tests and motor coordination assessments.
5. **Explainable AI (XAI)**: Visual 3D GradCAM++ heatmaps and SHAP feature attribution to establish clinical transparency.

---

## ✨ Key Features

| Capability | Technical Implementation & Description |
|---|---|
| 🧠 **Volumetric MRI Feature Extraction** | 3D ResNet-50 backbone powered by **MONAI** with Med3D transfer learning to extract high-dimensional neuroanatomical representations from NIfTI volumes (`.nii` / `.nii.gz`). |
| 🖼️ **Single-Slice & Image-Based Inference** | Adaptive 2D CNN feature extractor supporting axial MRI slice uploads (`.png`, `.jpg`) integrated with digital assessment scores. |
| 📈 **Longitudinal Sequence Modeling** | Bidirectional LSTM (**Bi-LSTM**) capturing temporal trajectory trends across serial clinical visits. |
| 🔀 **Cross-Modal Attention Fusion** | 8-head Multi-Head Cross-Attention (MHA) mechanism projecting imaging and clinical embeddings into a unified 256-dimensional patient representation. |
| 🎯 **Three-Class HD Staging** | Multi-class classification: **Pre-manifest**, **Early**, and **Advanced** stages with calibrated probability distributions. |
| ⏳ **Progression Forecasting** | Predicts quantitative **12-month and 24-month** UHDRS motor score change ($\Delta$) and assigns risk stratification (Low / Medium / High). |
| 🔥 **3D GradCAM++ Explainability** | Generates 3D voxel activation heatmaps overlaid onto axial MRI slices to localize regions driving stage predictions (caudate nucleus, putamen, lateral ventricles). |
| 📊 **SHAP Feature Attribution** | SHapley Additive exPlanations waterfall plots explaining the impact and directionality of clinical and genetic factors (CAG repeat, motor scores, age, TFC). |
| 🧪 **Digital Cognitive Testing Battery** | 5 standardized clinical tests: Symbol Digit Modalities Test (**SDMT**), **2-Back** Working Memory, **Verbal Fluency**, **Trail Making (A & B)**, and **Delayed Recall**. |
| ⚡ **Digital Motor Assessment** | Browser-based motor testing measuring finger tapping cadence, fine reaction speed, coordination stability, and tremor variance. |
| 🌐 **Interactive 3D Brain Visualizer** | Interactive WebGL / **Three.js** 3D brain mesh with region-of-interest highlighting and axial plane inspection. |
| 🤖 **Groq-Powered Clinical AI Assistant** | Low-latency LLM assistant conditioned on Mayo Clinic & NIH/NINDS clinical standards for HD pathology, pharmacotherapy, and platform queries. |
| 🗄️ **MongoDB Clinical Persistence** | Asynchronous database storage (Motor/PyMongo) tracking patient profiles, test logs, longitudinal assessments, and diagnostic histories. |
| 🌍 **Multi-Language Dashboard** | Dark-mode clinical UI built in React 19 with glassmorphism design, role-based workflows (Doctor & Patient), and real-time translation support. |

---

## 🏗️ System Architecture

```
                                 ┌─────────────────────────────────────────────────────┐
                                 │                   PATIENT DATA                      │
                                 └──────────┬───────────────────────────────┬──────────┘
                                            │                               │
                                            ▼                               ▼
                             ┌──────────────────────────────┐ ┌───────────────────────────┐
                             │    3D Structural MRI Scan    │ │   Clinical & Biomarkers   │
                             │        (.nii / .nii.gz)      │ │   (CAG, UHDRS, TFC, Age)  │
                             └──────────────┬───────────────┘ └─────────────┬─────────────┘
                                            │                               │
                                            ▼                               ▼
                             ┌──────────────────────────────┐ ┌───────────────────────────┐
                             │       3D ResNet-50           │ │         Bi-LSTM           │
                             │      MRI Volume Encoder      │ │     Clinical Encoder      │
                             │       (MONAI / Med3D)        │ │  (Temporal Visit Sequence)│
                             └──────────────┬───────────────┘ └─────────────┬─────────────┘
                                            │ [256-dim]                     │ [256-dim]
                                            └───────────────┬───────────────┘
                                                            ▼
                                            ┌───────────────────────────────┐
                                            │  Cross-Modal Attention Fusion │
                                            │        (8-head Attention)     │
                                            └───────────────┬───────────────┘
                                                            ▼
                                            ┌───────────────────────────────┐
                                            │   Fused Latent Representation │
                                            │            (256-dim)          │
                                            └───────────────┬───────────────┘
                                                            │
                     ┌──────────────────────────────────────┼──────────────────────────────────────┐
                     ▼                                      ▼                                      ▼
      ┌──────────────────────────────┐       ┌──────────────────────────────┐       ┌──────────────────────────────┐
      │     HD Stage Classifier      │       │    Progression Forecaster    │       │     Explainability Engine    │
      │   (Pre-manifest/Early/Adv)   │       │     (12 & 24 Month Delta)    │       │     (GradCAM++ & SHAP)       │
      └──────────────┬───────────────┘       └──────────────┬───────────────┘       └──────────────┬───────────────┘
                     │                                      │                                      │
                     └──────────────────────────────────────┼──────────────────────────────────────┘
                                                            ▼
                                             ┌─────────────────────────────┐
                                             │      FastAPI REST API       │
                                             │  (Asynchronous / Uvicorn)   │
                                             └──────────────┬──────────────┘
                                                            ▼
                                             ┌─────────────────────────────┐
                                             │   React 19 Dashboard & UI   │
                                             │ (3D Brain, Tests, Chatbot)  │
                                             └─────────────────────────────┘
```

---

## 📁 Repository Structure

```text
NeuroSense/
├── package.json                   # Root scripts for frontend orchestration
├── README.md                      # Project documentation and guide
├── .env.example                   # Environment variable template
├── exp.txt                        # Project abstract and concise summary
├── uploads/                       # Temporary upload staging directory
│   └── Execution.txt              # Quick execution cheatsheet
├── outputs/                       # Generated heatmaps and analysis reports
│   └── heatmaps/                  # GradCAM++ visualization artifacts
├── checkpoints/                   # Trained deep learning model weights
│   └── best_model.pt              # Production model checkpoint
├── neurosense/                    # Core Python application package
│   ├── api/                       # FastAPI backend service
│   │   ├── main.py                # Main application, lifespan & routing
│   │   ├── schemas.py             # Pydantic request/response schemas
│   │   ├── inference.py           # Deep learning inference pipeline
│   │   ├── clinical_scoring.py    # Clinical score calculation logic
│   │   └── routes/                # Modular sub-routers
│   │       ├── chatbot.py         # Groq AI clinical chatbot endpoints
│   │       └── cognitive.py       # Digital cognitive battery API
│   ├── cognitive/                 # Neuropsychological testing suite
│   │   ├── tests.py               # SDMT, N-Back, Fluency, Trails, Recall
│   │   ├── session.py             # Cognitive session & state management
│   │   └── scoring.py             # Normative z-scores & classification
│   ├── database/                  # MongoDB async database layer
│   │   ├── connection.py          # Motor connection manager & ping
│   │   ├── models.py              # User, prediction, and test models
│   │   └── crud.py                # Database queries and persistence
│   ├── models/                    # Neural network architectures
│   │   ├── mri_encoder.py         # 3D ResNet-50 volumetric encoder
│   │   ├── clinical_encoder.py    # Bi-LSTM longitudinal encoder
│   │   ├── fusion.py              # Cross-modal multi-head attention
│   │   └── classifier.py          # Staging and progression heads
│   ├── explainability/            # Interpretability & XAI modules
│   │   ├── gradcam.py             # 3D GradCAM++ implementation
│   │   ├── shap_analysis.py       # SHAP DeepExplainer integration
│   │   └── visualise.py           # Slice slicing and overlay generation
│   ├── training/                  # Model training and ablation pipeline
│   │   ├── train.py               # End-to-end training loop
│   │   ├── ablation.py            # Multi-condition ablation framework
│   │   ├── evaluate.py            # AUC, F1, and ECE evaluation metrics
│   │   └── losses.py              # Weighted cross-entropy & Huber loss
│   ├── data/                      # Dataset handling & neuroimaging transforms
│   │   ├── dataset.py             # MONAI dataset loaders
│   │   ├── preprocessing.py       # Spatial resampling, normalization
│   │   └── harmonizer.py          # Multi-site harmonization utilities
│   ├── configs/                   # Hyperparameter and model YAML configs
│   ├── scripts/                   # Setup and preprocessing automation
│   ├── tests/                     # Unit and integration test suite
│   └── frontend/                  # React clinical dashboard
│       ├── package.json           # Frontend dependencies & Vite scripts
│       ├── vite.config.js         # Vite build configuration
│       └── src/
│           ├── App.jsx            # Main app component & routing
│           ├── App.css            # Custom CSS & design system
│           ├── BrainViewer3D.jsx  # Three.js 3D brain visualizer
│           ├── LanguageContext.jsx# Multi-language context provider
│           ├── ToastContext.jsx   # Clinical notifications provider
│           └── translations.js    # Internationalization dictionary
```

---

## 🛠️ Technology Stack

| Layer | Tools & Libraries |
|---|---|
| **Deep Learning** | PyTorch 2.x, MONAI 1.3, Torchvision |
| **Neuroimaging I/O** | NiBabel, FSL, ANTs |
| **Explainable AI (XAI)** | PyTorch-Grad-CAM, SHAP |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **Database** | MongoDB, Motor (Async driver), PyMongo |
| **AI Chatbot** | Groq Cloud API (`openai/gpt-oss-120b`, `qwen/qwen3.6-27b`) |
| **Frontend Framework** | React 19, Vite 8, React Router v7 |
| **Data Viz & 3D** | Three.js (3D Brain), Recharts, Framer Motion |
| **Language & Env** | Python 3.10+, Node.js 18+, CUDA 11.8+ |

---

## 🚀 Quickstart Guide

### 1. Prerequisites

Ensure you have installed:
- **Python**: version `3.10` or higher
- **Node.js**: version `18` or higher and `npm`
- **MongoDB**: local instance running on `localhost:27017` (or remote MongoDB URI)
- *(Optional)* **NVIDIA GPU** with CUDA 11.8+ for accelerated inference/training

---

### 2. Environment Configuration

Copy the example environment template and configure your parameters:

```bash
cp .env.example .env
```

Edit `.env` to supply your configuration:
```env
# MongoDB Database Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=neurosense

# Groq AI Chatbot API Configuration (Optional: for AI Assistant)
GROQ_API_KEY=your_groq_api_key_here
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
GROQ_MODEL=openai/gpt-oss-120b
```

---

### 3. Backend Setup

Create and activate a Python virtual environment, then install all required packages:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r neurosense/requirements.txt
```

---

### 4. Frontend Setup

Install the React client dependencies:

```bash
# Using root shortcut
npm install --prefix neurosense/frontend

# Or directly in the frontend directory:
# cd neurosense/frontend && npm install
```

---

### 5. Running the Application

You can launch both servers simultaneously in separate terminals:

#### Terminal 1 — Start FastAPI Server
```bash
# From workspace root:
uvicorn neurosense.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Interactive OpenAPI / Swagger Docs: `http://localhost:8000/docs`
- Health Status: `http://localhost:8000/health`

#### Terminal 2 — Start React Dashboard
```bash
# From workspace root:
npm run dev
```
- Frontend Application URL: `http://localhost:5173`

---

## 📡 API Reference Overview

The FastAPI service exposes a comprehensive suite of REST endpoints:

### Prediction & Diagnostic Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Full multi-modal prediction using a 3D MRI volume (`.nii`/`.nii.gz`) + clinical tabular inputs. Returns HD stage, progression forecast, 3D GradCAM++ heatmap link, and SHAP attributions. |
| `POST` | `/predict-image` | High-speed prediction using a 2D MRI slice (`.png`/`.jpg`) combined with digital cognitive/motor assessment percentages and CAG count. |

### Digital Cognitive Battery Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/cognitive/session/start` | Initializes a new randomized assessment session with parallel-form version rotation. |
| `POST` | `/cognitive/session/submit` | Submits individual test responses (SDMT, N-Back, Fluency, Trails, Recall). |
| `POST` | `/cognitive/session/complete` | Calculates normative z-scores, domain percentiles, and aggregates a composite cognitive index. |
| `GET` | `/cognitive/patient/{id}/history` | Retrieves patient longitudinal cognitive trajectory and score deltas. |

### Clinical Chatbot & Assistant

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chatbot/chat` | Groq-accelerated clinical assistant trained on HD symptomatology, Mayo Clinic standards, and platform usage. |

### User, Patient & History Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/users` | Registers a new patient or clinician account. |
| `POST` | `/login` | Authenticates clinician or patient credentials. |
| `GET` | `/patients` | Lists all registered patients for clinician assignment and tracking. |
| `GET` | `/users/{id}/predictions` | Returns complete historical predictions and trajectories for a patient. |
| `POST` | `/users/{id}/tests` | Records completed digital motor or cognitive test results. |
| `GET` | `/health` | Reports service health, uptime, and GPU acceleration status. |

---

## 🧠 Digital Assessment Battery Details

### 1. Neuropsychological Tests (`/memory-test`)
- **Symbol Digit Modalities Test (SDMT)**: Evaluates complex visual scanning, information processing speed, and motor persistence.
- **2-Back Working Memory Task**: Measures working memory updating and executive working memory capacity under cognitive load.
- **Phonemic Verbal Fluency**: Assesses lexical retrieval speed, phonemic search, and frontal lobe executive function.
- **Trail Making Test (Parts A & B)**: Quantifies visuomotor tracking speed (Part A) and executive task switching / cognitive flexibility (Part B).
- **Delayed Word Recall**: Tests short-term retention, episodic memory consolidation, and delayed retrieval mechanisms.

### 2. Motor Coordination Battery (`/motor-test`)
- **Finger Tapping Cadence**: Quantifies inter-tap interval regularity, tapping frequency, and motor persistence to capture subtle bradykinesia.
- **Reaction Time & Latency**: Tests visual stimulus detection speed and motor response latency.
- **Motor Cadence & Stability**: Tracks rhythm stability and movement irregularity indicative of early choreic interference.

---

## 🧪 Model Performance & Ablation Benchmarks

During multi-modal ablation validation across 5 configurations (evaluated with 3 distinct random seeds), the cross-modal attention fusion mechanism achieved superior diagnostic accuracy:

| Configuration | 3D MRI Input | Clinical & Genetic Metrics | Stage AUC (Mean ± Std) | 12-Month Huber Loss |
|---|:---:|:---:|:---:|:---:|
| **MRI Only (Baseline)** | ✅ | ❌ | $0.803 \pm 0.015$ | $3.42$ |
| **Clinical Only** | ❌ | ✅ | $0.761 \pm 0.018$ | $3.89$ |
| **Genetic Only (CAG)** | ❌ | CAG Only | $0.724 \pm 0.021$ | $4.25$ |
| **MRI + Clinical (Feature Concat)** | ✅ | ✅ | $0.838 \pm 0.012$ | $3.15$ |
| **NeuroSense (Cross-Attention Fusion)** | ✅ | ✅ | **$0.887 \pm 0.009$** | **$2.68$** |

*Cross-modal attention dynamically weights MRI volumetric signals against genetic and clinical progression markers, preventing false positives in early pre-manifest stages.*

---

## 🔒 Security & Privacy

- **De-Identification**: Designed to ingest de-faced and de-identified structural neuroimaging adhering to HIPAA Safe Harbor standards.
- **Transient Processing**: Uploaded raw NIfTI files are processed in isolated temporary workspaces and automatically unlinked after feature extraction.
- **Role Isolation**: Doctor-only routes ensure clinical tests and raw diagnostic parameters are administered under supervised clinical protocols.

---

## 📄 License & Attribution

This project is licensed for **academic, educational, and research use**. Datasets utilized for training and validation (such as OpenNeuro ds004040 and PREDICT-HD) remain subject to their respective Data Use Agreements (DUA) and FITBIR governance.

**Principal Author**: Suraj S  
**Project**: NeuroSense — Huntington's Disease AI Staging & Progression Platform