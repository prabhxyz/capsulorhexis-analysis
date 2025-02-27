#!/usr/bin/env bash
#
# Usage:
#   ./train_segmentation.sh /path/to/Cataract-1k-Seg
#
# Description:
#   1) Installs dependencies if needed.
#   2) Runs the segmentation training script with chosen hyperparameters.
#
# Example:
#   ./train_segmentation.sh Cataract-1k-Seg
#

ROOT_DIR=$1

if [ -z "$ROOT_DIR" ]; then
  echo "Usage: $0 /path/to/Cataract-1k-Seg"
  exit 1
fi

echo "Training segmentation model on dataset at: $ROOT_DIR"

# Hyperparameters 
EPOCHS=12
BATCH_SIZE=4
LR=2e-4

# Install requirements:
pip install -r requirements.txt

python train_segmentation.py \
  --seg_data_root $ROOT_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR
