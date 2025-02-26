import os
import argparse
import torch
import cv2
import numpy as np

from phase_recognition.model_phase import PhaseRecognitionModel
from phase_recognition.utils_phase import accuracy_score

from segmentation.model_seg import get_mask_rcnn_model
from capsule_analysis.detect_capsule_math import track_instrument_and_detect_capsule
from capsule_analysis.measure_rhexis import measure_rhexis
from capsule_analysis.classify_risk import classify_rhexis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video to analyze.")
    parser.add_argument("--phase_model_dir", type=str, required=True, help="Folder containing phase_recognition_swin3d.pth.")
    parser.add_argument("--seg_model_dir", type=str, required=True, help="Folder containing mask_rcnn_cataract.pth.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load Phase Recognition Model
    phase_ckpt = os.path.join(args.phase_model_dir, "phase_recognition_swin3d.pth")
    # For simplicity, let's guess we have 10 phases
    num_phases = 10  # or the correct number used during training
    phase_model = PhaseRecognitionModel(num_phases=num_phases)
    phase_model.load_state_dict(torch.load(phase_ckpt, map_location=device))
    phase_model.to(device)
    phase_model.eval()

    # For a real approach, one needs a clip-based approach to find the capsulorhexis segment.
    # Here, for demonstration, we'll simply read the entire video into memory (not feasible for large videos).
    # Then we can do a naive approach or a more advanced sliding window to find the "capsulorhexis" frames.

    cap = cv2.VideoCapture(args.video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    # Let's pretend we have the entire video. We'll do a quick "dummy" approach to find the segment around frames 0.2 to 0.3 of the video as "capsulorhexis"
    # In a real scenario, one would pass short clips to the phase model. We'll skip that for brevity.

    start_idx = int(len(frames)*0.2)
    end_idx = int(len(frames)*0.3)
    capsulorhexis_frames = frames[start_idx:end_idx]

    # 2) Load segmentation model
    # The relevant classes + 1 for background
    num_relevant_classes = 4  # 3 instruments + background
    seg_model = get_mask_rcnn_model(num_relevant_classes)
    seg_ckpt = os.path.join(args.seg_model_dir, "mask_rcnn_cataract.pth")
    seg_model.load_state_dict(torch.load(seg_ckpt, map_location=device))
    seg_model.to(device)
    seg_model.eval()

    # 3) Detect final capsulorhexis
    # We'll track the instrument and see if we get a circle
    final_mask = track_instrument_and_detect_capsule(capsulorhexis_frames, seg_model, device, threshold=0.5)

    if final_mask is None:
        print("No complete capsulorhexis detected. Possibly At-Risk due to incomplete tear.")
        return

    # 4) Measure rhexis
    rhexis_info = measure_rhexis(final_mask, reference_instrument_size=1.5)

    # 5) Classify risk
    label, reasons = classify_rhexis(rhexis_info,
                                     center_threshold=1.0,
                                     ideal_diameter_range=(4.5, 5.5),
                                     circularity_threshold=0.8,
                                     eye_center=(256,256))  # naive guess

    print("===== CAPSULORHEXIS EVALUATION =====")
    print(f"Diameter (mm): {rhexis_info['diameter_mm']:.2f}")
    print(f"Circularity: {rhexis_info['circularity']:.2f}")
    print(f"Center (px): {rhexis_info['center']}")
    print(f"Risk Classification: {label}")
    if label == "At-Risk":
        print("Reasons:")
        for r in reasons:
            print(f" - {r}")

if __name__ == "__main__":
    main()