#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# NeuroSense — Preprocessing Pipeline
# PRD Section 4.2.1: Preprocessing Module
# Usage: bash scripts/preprocess_all.sh [--input-dir PATH] [--output-dir PATH]
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ─── Defaults ───
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/processed}"

# ─── Parse arguments ───
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)  INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --help)
            echo "Usage: bash scripts/preprocess_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-dir PATH    Raw data directory (default: data/raw)"
            echo "  --output-dir PATH   Output directory (default: data/processed)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          NeuroSense — MRI Preprocessing Pipeline        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# ─── Activate virtual environment if available ───
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# ─── Run preprocessing via Python module ───
# This calls neurosense.data.preprocessing which handles:
#   1. Skull stripping (FSL BET or MONAI fallback)
#   2. Bias field correction
#   3. Affine registration to MNI152
#   4. Resampling to 1mm isotropic → 96×96×96
#   5. Z-score normalisation per subject
#   6. Tabular feature normalisation (min-max + median imputation)

echo "→ Running MRI preprocessing pipeline ..."
echo "  Pipeline: skull strip → bias correction → MNI152 registration"
echo "            → resample 96³ → z-score normalisation"
echo ""

python3 -m neurosense.data.preprocessing \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --spatial-size 96 96 96 \
    --voxel-spacing 1.0 1.0 1.0 \
    --normalize zscore \
    --num-workers 4

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ Preprocessing complete!                             ║"
echo "║                                                         ║"
echo "║  Processed data saved to:                               ║"
echo "║    $OUTPUT_DIR                                          ║"
echo "║                                                         ║"
echo "║  Next: python -m neurosense.training.train              ║"
echo "╚══════════════════════════════════════════════════════════╝"
