#!/usr/bin/env bash
#
# Usage:
#   chmod +x inference_pipeline.sh
#   ./inference_pipeline.sh /path/to/video.mp4 /path/to/phase_recognition.pth /path/to/lightweight_seg.pth
#
# Description:
#   Runs the entire inference pipeline on a single video using the pretrained
#   phase recognition model and segmentation model. Focuses on the capsulorhexis phase.
#

VIDEO_PATH=$1
PHASE_MODEL=$2
SEG_MODEL=$3
FRAME_STEP=$4

if [ -z "$VIDEO_PATH" ] || [ -z "$PHASE_MODEL" ] || [ -z "$SEG_MODEL" ] || [ -z "$FRAME_STEP" ]; then
  echo "Usage: $0 /path/to/video.mp4 /path/to/phase_model.pth /path/to/seg_model.pth capsulorhexis_phase_idx"
  exit 1
fi

echo "Running inference on $VIDEO_PATH"
echo "Phase model: $PHASE_MODEL"
echo "Segmentation model: $SEG_MODEL"
echo "Capsulorhexis Phase Index: $FRAME_STEP"

# Install requirements:
echo "Installing required packages..."
pip install -r requirements.txt

# Install OpenGL library (libGL.so.1)
echo "Installing libGL.so.1 (OpenGL)..."
#sudo apt update
sudo apt install -y libgl1-mesa-glx

python pipeline_infer.py \
  --video_path "$VIDEO_PATH" \
  --phase_model_path "$PHASE_MODEL" \
  --seg_model_path "$SEG_MODEL" \
  --caps_phase_idx $FRAME_STEP \

echo "Inference complete."