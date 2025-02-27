#!/usr/bin/env bash
# Usage: ./train_both.sh /path/to/Cataract-1k-Phase /path/to/Cataract-1k-Seg

PHASE_ROOT=$1
SEG_ROOT=$2
if [ -z "$PHASE_ROOT" ] || [ -z "$SEG_ROOT" ]; then
  echo "Usage: ./train_both.sh <phase_root> <seg_root>"
  exit 1
fi

python ./training/train_both.py \
    --phase_root "$PHASE_ROOT" \
    --seg_root "$SEG_ROOT" \
    --phase_epochs 10 \
    --seg_epochs 10 \
    --batch_size_phase 8 \
    --batch_size_seg 6 \
    --lr_phase 1e-4 \
    --lr_seg 1e-4
