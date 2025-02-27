#!/usr/bin/env bash
# Usage: ./train_segmentation.sh /path/to/Cataract-1k-Seg

SEG_ROOT=$1
if [ -z "$SEG_ROOT" ]; then
  echo "Usage: ./train_segmentation.sh <seg_root>"
  exit 1
fi

python ./training/train_segmentation.py --root_dir "$SEG_ROOT" \
    --seg_epochs 10 \
    --batch_size 6 \
    --lr 1e-4
