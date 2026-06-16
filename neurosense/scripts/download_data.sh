#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# NeuroSense — Dataset Download Script
# PRD Section 3: Datasets
# Usage: bash scripts/download_data.sh [--openneuro | --predict-hd]
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/raw"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          NeuroSense — Dataset Download                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Parse arguments ───
DATASET="${1:---openneuro}"

case "$DATASET" in
    --openneuro)
        echo "→ Downloading OpenNeuro ds004040 (supplementary HD dataset)"
        echo "  Source: https://openneuro.org/datasets/ds004040"
        echo "  This dataset is publicly available — no approval required."
        echo ""

        OPENNEURO_DIR="$DATA_DIR/openneuro_ds004040"
        mkdir -p "$OPENNEURO_DIR"

        # Check if AWS CLI or datalad is available
        if command -v aws &> /dev/null; then
            echo "→ Using AWS CLI for download ..."
            aws s3 sync --no-sign-request \
                s3://openneuro.org/ds004040 \
                "$OPENNEURO_DIR" \
                --exclude "derivatives/*"
            echo "✓ Download complete via AWS CLI"

        elif command -v datalad &> /dev/null; then
            echo "→ Using DataLad for download ..."
            datalad install \
                "https://github.com/OpenNeuroDatasets/ds004040.git" \
                "$OPENNEURO_DIR"
            cd "$OPENNEURO_DIR"
            datalad get -r .
            echo "✓ Download complete via DataLad"

        else
            echo "⚠ Neither AWS CLI nor DataLad found."
            echo "  Install one of:"
            echo "    pip install awscli"
            echo "    pip install datalad"
            echo ""
            echo "  Or download manually from:"
            echo "    https://openneuro.org/datasets/ds004040"
            echo ""
            echo "  Place BIDS-format data in:"
            echo "    $OPENNEURO_DIR"
            exit 1
        fi

        echo ""
        echo "✓ Data saved to: $OPENNEURO_DIR"
        echo "  Update configs/train_config.yaml → data.root_dir"
        ;;

    --predict-hd)
        echo "→ PREDICT-HD Dataset (Primary)"
        echo "  Source: FITBIR — https://fitbir.nih.gov"
        echo ""
        echo "  ⚠ PREDICT-HD requires a data access agreement."
        echo "    1. Register at https://fitbir.nih.gov"
        echo "    2. Submit a Data Access Request for PREDICT-HD"
        echo "    3. Allow 2–4 weeks for approval"
        echo ""
        echo "  Once approved, download the data and place it in:"
        echo "    $DATA_DIR/predict_hd/"
        echo ""
        echo "  Expected BIDS structure:"
        echo "    predict_hd/"
        echo "    ├── participants.tsv"
        echo "    ├── sub-001/"
        echo "    │   ├── anat/"
        echo "    │   │   └── sub-001_T1w.nii.gz"
        echo "    │   └── phenotype/"
        echo "    │       └── clinical.tsv"
        echo "    ├── sub-002/"
        echo "    └── ..."

        mkdir -p "$DATA_DIR/predict_hd"
        echo ""
        echo "✓ Directory created: $DATA_DIR/predict_hd/"
        ;;

    *)
        echo "Usage: bash scripts/download_data.sh [--openneuro | --predict-hd]"
        echo ""
        echo "  --openneuro    Download OpenNeuro ds004040 (public, no approval)"
        echo "  --predict-hd   Instructions for PREDICT-HD (requires FITBIR approval)"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Next: bash scripts/preprocess_all.sh                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
