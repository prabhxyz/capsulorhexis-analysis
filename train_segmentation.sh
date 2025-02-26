#!/usr/bin/env bash
# Example usage: ./train_segmentation.sh /path/to/Cataract-1k-Seg

ROOT_DIR=$1
python segmentation/train_seg.py --root_dir $ROOT_DIR
