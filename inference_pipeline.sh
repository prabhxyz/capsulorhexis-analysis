#!/usr/bin/env bash
# Example usage: ./inference_pipeline.sh /path/to/video.mp4 /path/to/models/phase_recognition /path/to/models/segmentation

VIDEO_PATH=$1
PHASE_MODEL_DIR=$2
SEG_MODEL_DIR=$3

python pipeline_infer.py --video_path $VIDEO_PATH \
                         --phase_model_dir $PHASE_MODEL_DIR \
                         --seg_model_dir $SEG_MODEL_DIR
