#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# NeuroSense — Environment Setup Script
# PRD NFR-07: Deployable with a single setup script
# Usage: bash scripts/setup_env.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          NeuroSense — Environment Setup                 ║"
echo "║  AI-Powered Huntington's Disease Detection Platform     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Check Python version ───
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    echo "   Install via: conda install python=3.10 or pyenv install 3.10"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION detected"

# ─── Create virtual environment ───
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# ─── Activate and install dependencies ───
echo "→ Installing Python dependencies ..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r "$PROJECT_ROOT/requirements.txt"
echo "✓ Python dependencies installed"

# ─── Check CUDA availability ───
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    print(f'✓ CUDA available: {gpu}')
    print(f'  CUDA version: {torch.version.cuda}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ Apple MPS (Metal) available — GPU acceleration enabled')
else:
    print('⚠ No GPU detected — training will use CPU (significantly slower)')
    print('  For CUDA: install torch with CUDA support')
    print('  For MPS:  macOS 12.3+ with Apple Silicon required')
"

# ─── Create required directories ───
echo "→ Creating project directories ..."
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/checkpoints"
mkdir -p "$PROJECT_ROOT/outputs/heatmaps"
mkdir -p "$PROJECT_ROOT/outputs/reports"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/notebooks"
echo "✓ Project directories created"

# ─── Verify installation ───
echo ""
echo "→ Verifying critical imports ..."
python3 -c "
import torch
import monai
import nibabel
import fastapi
import sklearn
import shap
print('✓ All critical packages imported successfully')
print(f'  PyTorch:  {torch.__version__}')
print(f'  MONAI:    {monai.__version__}')
print(f'  NiBabel:  {nibabel.__version__}')
print(f'  FastAPI:  {fastapi.__version__}')
print(f'  sklearn:  {sklearn.__version__}')
"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup complete!                                     ║"
echo "║                                                         ║"
echo "║  Activate environment:                                  ║"
echo "║    source venv/bin/activate                              ║"
echo "║                                                         ║"
echo "║  Next steps:                                            ║"
echo "║    1. Download data:  bash scripts/download_data.sh     ║"
echo "║    2. Preprocess:     bash scripts/preprocess_all.sh    ║"
echo "║    3. Train:          python -m neurosense.training.train║"
echo "╚══════════════════════════════════════════════════════════╝"
