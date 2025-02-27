#!/usr/bin/env bash
# Usage: ./train_phase.sh /path/to/Cataract-1k-Phase

PHASE_ROOT=$1
if [ -z "$PHASE_ROOT" ]; then
  echo "Usage: ./train_phase.sh <phase_root>"
  exit 1
fi

python ./training/train_phase.py --root_dir "$PHASE_ROOT" \
    --phase_epochs 10 \
    --batch_size 8 \
    --lr 1e-4