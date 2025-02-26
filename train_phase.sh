#!/usr/bin/env bash
# Example usage: ./train_phase.sh /path/to/Cataract-1k-Phase

ROOT_DIR=$1
python phase_recognition/train_phase.py --root_dir $ROOT_DIR
