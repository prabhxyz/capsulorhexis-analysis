#!/usr/bin/env bash
# Trains both the Phase Recognition model and the Segmentation model in one script.

# Usage:
#   chmod +x train_both.sh
#   ./train_both.sh /path/to/Cataract-1k-Phase /path/to/Cataract-1k-Seg

PHASE_DATA_ROOT=$1
SEG_DATA_ROOT=$2

echo "=== TRAINING BOTH MODELS ==="
echo "Phase data root: $PHASE_DATA_ROOT"
echo "Seg data root:   $SEG_DATA_ROOT"

# Hard-code some hyperparameters (can be changed):
PHASE_EPOCHS=10
PHASE_BATCH=8
PHASE_LR=1e-4

SEG_EPOCHS=10
SEG_BATCH=4
SEG_LR=1e-4

# Call train_both.py, passing these arguments
python ./training/train_both.py \
    --phase_root "$PHASE_DATA_ROOT" \
    --phase_epochs $PHASE_EPOCHS \
    --phase_batch $PHASE_BATCH \
    --phase_lr $PHASE_LR \
    --seg_root "$SEG_DATA_ROOT" \
    --seg_epochs $SEG_EPOCHS \
    --seg_batch $SEG_BATCH \
    --seg_lr $SEG_LR
