import os
import cv2
import torch
import numpy as np
import pandas as pd
import argparse

from model import get_segmentation_model
from dataset import get_val_transforms
from measure_capsulorhexis import measure_capsulorhexis_region

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="unet")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b0")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--model_path", type=str, default="best_segmentation_model.pth")
    parser.add_argument("--frames_dir", type=str, default="/cataract-phase/temp/")
    parser.add_argument("--output_dir", type=str, default="inference_results/")
    parser.add_argument("--forceps_class_id", type=int, default=1,
                        help="Which label ID is the known-size forceps (1?).")
    parser.add_argument("--rhexis_class_id", type=int, default=2,
                        help="Which label ID is the capsulorhexis boundary (2?).")
    parser.add_argument("--known_forceps_size_mm", type=float, default=2.0)
    parser.add_argument("--save_overlay", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_segmentation_model(args.model_type, args.num_classes, args.encoder_name)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    transforms = get_val_transforms()

    results = []
    frame_files = sorted([
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    for frame_file in frame_files:
        frame_path = os.path.join(args.frames_dir, frame_file)
        original_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if original_img is None:
            continue

        H_orig, W_orig = original_img.shape[:2]
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        aug = transforms(image=img_rgb)
        input_tensor = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Upsample back to original size
        pred_upsampled = cv2.resize(pred.astype(np.uint8), (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

        # Forceps & Capsulorhexis masks
        forceps_mask = (pred_upsampled == args.forceps_class_id).astype(np.uint8)
        rhexis_mask = (pred_upsampled == args.rhexis_class_id).astype(np.uint8)

        # measure forceps => mm/pixel
        forceps_contours, _ = cv2.findContours(forceps_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(forceps_contours)>0:
            largest_forceps = max(forceps_contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(largest_forceps)
            forceps_width_px = max(w,h)
            mm_per_pixel = (args.known_forceps_size_mm / float(forceps_width_px)) if forceps_width_px>0 else 0.0
        else:
            mm_per_pixel = 0.0

        diameter_px, circ, offset_px = measure_capsulorhexis_region(rhexis_mask)
        diameter_mm = diameter_px * mm_per_pixel

        frame_id = os.path.splitext(frame_file)[0]
        results.append([frame_id, diameter_mm, circ, offset_px])

        # optional overlay
        if args.save_overlay:
            overlay = original_img.copy()
            # draw forceps in green
            for cnt in forceps_contours:
                cv2.drawContours(overlay, [cnt], -1, (0,255,0), 2)
            # draw rhexis in red
            rhexis_contours, _ = cv2.findContours(rhexis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in rhexis_contours:
                cv2.drawContours(overlay, [cnt], -1, (0,0,255), 2)
            out_path = os.path.join(args.output_dir, f"{frame_id}_overlay.jpg")
            cv2.imwrite(out_path, overlay)

    # Save CSV
    df = pd.DataFrame(results, columns=["frame_id","diameter_mm","circularity","center_offset_px"])
    csv_path = os.path.join(args.output_dir, "capsulorhexis_measurements.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results => {csv_path}")

if __name__ == "__main__":
    main()
