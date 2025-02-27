#!/usr/bin/env bash
# Usage: ./inference_pipeline.sh <video_path> <phase_model> <seg_model>
VIDEO_PATH=$1
PHASE_MODEL=$2
SEG_MODEL=$3

echo "Running inference on $VIDEO_PATH"
echo "Phase model: $PHASE_MODEL"
echo "Segmentation model: $SEG_MODEL"

python pipeline_infer.py \
    --video_path "$VIDEO_PATH" \
    --phase_model_path "$PHASE_MODEL" \
    --seg_model_path "$SEG_MODEL"

echo "Inference complete."