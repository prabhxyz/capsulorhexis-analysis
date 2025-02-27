#!/usr/bin/env bash
#
# Usage:
#   ./inference_pipeline.sh /path/to/video.mp4 /path/to/phase_recognition.pth /path/to/lightweight_seg.pth
#
# Description:
#   Runs the entire inference pipeline on a single video using the pretrained
#   phase recognition model and segmentation model. Focuses on the capsulorhexis phase.
#

VIDEO_PATH=$1
PHASE_MODEL=$2
SEG_MODEL=$3

if [ -z "$VIDEO_PATH" ] || [ -z "$PHASE_MODEL" ] || [ -z "$SEG_MODEL" ]; then
  echo "Usage: $0 /path/to/video.mp4 /path/to/phase_model.pth /path/to/seg_model.pth"
  exit 1
fi

echo "Running inference on $VIDEO_PATH"
echo "Phase model: $PHASE_MODEL"
echo "Segmentation model: $SEG_MODEL"

python pipeline_infer.py \
  --video_path "$VIDEO_PATH" \
  --phase_model_path "$PHASE_MODEL" \
  --seg_model_path "$SEG_MODEL"

echo "Inference complete."
