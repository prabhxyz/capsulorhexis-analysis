import argparse
import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import deque

from models.phase_recognition_model import PhaseRecognitionNet
from models.segmentation_model import LightweightSegModel

# Suppose you have a geometry-based or any other module to detect rhexis, measure, etc.
# from capsule_analysis.detect_capsule_math import track_instrument_and_detect_capsule
# from capsule_analysis.measure_rhexis import measure_rhexis
# from capsule_analysis.classify_risk import classify_rhexis

###########################################################
# UTILS
###########################################################

def _sample_and_classify_frames(cap, phase_model, phase_transform, device,
                                sample_interval=20,
                                smoothing_window=15,
                                target_phase_idx=None):
    """
    Quickly scans the entire video by sampling every 'sample_interval' frames.
    Classifies each sampled frame with the pretrained single-frame model.

    smoothing_window: number of recent classifications to consider for smoothing
    target_phase_idx: the integer index for 'Capsulorhexis' if known

    Returns:
      A dictionary with:
        - raw_classifications: list of (frame_idx, predicted_phase_idx)
        - capsulorhexis_range: (start_frame, end_frame) if found, else None
    """
    # We'll store raw classification results:
    raw_classifications = []
    # We'll do a rolling queue to find consecutive "target_phase_idx" classification
    # with some tolerance for misclassifications
    frame_index = 0

    # This queue holds the last 'smoothing_window' predictions. We'll do a majority check.
    recent_preds = deque(maxlen=smoothing_window)
    capsulorhexis_start = None
    capsulorhexis_end   = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        # Move capture pointer
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Single-frame transform
        with torch.no_grad():
            inp = phase_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            logits = phase_model(inp)
            pred_idx = torch.argmax(logits, dim=1).item()

        raw_classifications.append((frame_index, pred_idx))

        # Add to rolling queue
        recent_preds.append(pred_idx)
        # Check majority
        if target_phase_idx is not None:
            # count how many times 'target_phase_idx' in recent queue
            count_target = sum(1 for x in recent_preds if x == target_phase_idx)
            # We'll define a threshold: if 75% of the last 'smoothing_window' frames match target_phase_idx,
            # we consider ourselves "in" the capsulorhexis phase
            if capsulorhexis_start is None:
                # If not started yet:
                if count_target >= 0.75 * len(recent_preds):
                    # We consider this the start
                    capsulorhexis_start = frame_index
            else:
                # If we've started, see if we've ended
                # If fewer than 25% match the target in recent frames, consider the phase ended
                if count_target < 0.25 * len(recent_preds):
                    capsulorhexis_end = frame_index
                    break  # We can exit early if we only want the first occurrence

        frame_index += sample_interval
        if frame_index >= total_frames:
            break

    # If we found a start but never found an end, we can set end as near last frame
    if capsulorhexis_start is not None and capsulorhexis_end is None:
        capsulorhexis_end = total_frames - 1

    # Reset capture pointer to 0 for next usage
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if capsulorhexis_start is not None:
        return {
            "raw_classifications": raw_classifications,
            "capsulorhexis_range": (capsulorhexis_start, capsulorhexis_end)
        }
    else:
        return {
            "raw_classifications": raw_classifications,
            "capsulorhexis_range": None
        }

###########################################################
# MAIN
###########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the cataract-surgery video.")
    parser.add_argument("--phase_model_path", type=str, default="phase_recognition.pth",
                        help="Path to the pretrained single-frame phase model.")
    parser.add_argument("--seg_model_path", type=str, default="lightweight_seg.pth",
                        help="Path to the segmentation model.")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--capsulorhexis_phase_idx", type=int, default=2,
                        help="Index representing the 'Capsulorhexis' label in the phase model. Adjust as needed.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 1) Load the pretrained single-frame phase model
    from models.phase_recognition_model import PhaseRecognitionNet
    # Suppose we have 12 phases total
    phase_model = PhaseRecognitionNet(num_phases=12, use_pretrained=False).to(device)
    phase_model.load_state_dict(torch.load(args.phase_model_path, map_location=device))
    phase_model.eval()

    # 2) Phase transform
    phase_transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # 3) Open the video and do a quick sampling to find the capsulorhexis segment
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error opening video: {args.video_path}")
        return

    info = _sample_and_classify_frames(
        cap,
        phase_model,
        phase_transform,
        device,
        sample_interval=20,       # only check every 20 frames
        smoothing_window=15,      # how many frames to consider for majority
        target_phase_idx=args.capsulorhexis_phase_idx
    )
    cap.release()

    caps_range = info["capsulorhexis_range"]
    if not caps_range:
        print("No reliable capsulorhexis phase detected. We'll proceed with entire video or skip.")
        caps_range = (0, None)  # fallback to entire video if needed

    start_frame, end_frame = caps_range
    print(f"Detected capsulorhexis from frame {start_frame} to {end_frame}.")

    # 4) Now let's do segmentation only on that portion
    # For demonstration, we'll read from start_frame to end_frame
    cap2 = cv2.VideoCapture(args.video_path)
    total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame >= total_frames:
        end_frame = total_frames - 1

    # 4a) Load the segmentation model
    from models.segmentation_model import LightweightSegModel
    seg_model = LightweightSegModel(num_classes=4, use_pretrained=False, aux_loss=True).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_path, map_location=device))
    seg_model.eval()

    seg_transform = A.Compose([
        A.Resize(512,512),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # We will process frames in the range [start_frame, end_frame]
    current_frame = start_frame
    while current_frame <= end_frame:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame_bgr = cap2.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            inp = seg_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            seg_out = seg_model(inp)["out"]
            seg_pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()
        # => seg_pred is the segmentation map for relevant frames
        # If you want to track instruments or detect rhexis geometry, do it here.

        # For demonstration, we just show the current frame index:
        print(f"Processed frame {current_frame} in the capsulorhexis phase range.")
        current_frame += 1

    cap2.release()
    print("Done inference focusing on capsulorhexis phase. Exiting.")

if __name__ == "__main__":
    main()