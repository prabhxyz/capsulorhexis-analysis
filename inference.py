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
    # model inputs
    parser.add_argument("--model_type", type=str, default="deeplab")
    parser.add_argument("--encoder_name", type=str, default="resnet101")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="fold_1/best_model.pth")
    parser.add_argument("--frames_dir", type=str, default="/cataract-phase/temp/")
    parser.add_argument("--output_dir", type=str, default="temporal_inference_results/")
    # relevant classes
    parser.add_argument("--forceps_class_id", type=int, default=2)
    parser.add_argument("--rhexis_class_id", type=int, default=3)
    parser.add_argument("--known_forceps_size_mm", type=float, default=2.0)
    # advanced:
    parser.add_argument("--union_strategy", type=str, default="union", 
                        choices=["union","intersection","last"],
                        help="How to merge rhexis masks over frames.")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--skip_threshold", type=float, default=50.0,
                        help="If instrument tip moves > skip_threshold px between frames, skip that frame as outlier.")
    parser.add_argument("--circularity_filter", type=float, default=0.3,
                        help="If final shape circularity < this, we suspect it's not a valid rhexis. Adjust as needed.")
    parser.add_argument("--register_frames", action="store_true",
                        help="If set, attempts a simple feature-based registration to stabilize eye movement.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_segmentation_model(args.model_type, args.num_classes, args.encoder_name)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    transforms = get_val_transforms()

    frame_files = sorted([
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith(('.jpg','.png','.jpeg'))
    ])

    # Prepare variables for:
    # 1) Accumulated rhexis mask
    # 2) Running scale factors
    # 3) Storing the tip position each frame => we can skip frames if movement is suspicious
    accumulated_rhexis = None
    scales = []
    last_tip_center = None  # track instrument tip from previous frame
    robust_rhexis_frames = []  # store (frame_index, rhexis_mask, mm_per_pixel)
    
    # For registration, store the first frame as reference
    reference_gray = None
    H_ref, W_ref = None, None
    homographies = []  # store transformations if we do frame registration

    orb = cv2.ORB_create()  # or use SIFT, etc. for feature-based alignment

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(args.frames_dir, frame_file)
        original_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if original_img is None:
            continue

        # Possibly convert for alignment
        if args.register_frames:
            current_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            if reference_gray is None:
                # first frame => reference
                reference_gray = current_gray
                H_ref, W_ref = original_img.shape[:2]
                # store identity transform
                homographies.append(np.eye(3))
            else:
                # find transformation from current_gray to reference_gray
                # naive approach: ORB keypoints, BF matching => find homography
                kp1, des1 = orb.detectAndCompute(reference_gray, None)
                kp2, des2 = orb.detectAndCompute(current_gray, None)
                if des1 is None or des2 is None or len(kp1)<4 or len(kp2)<4:
                    # fallback
                    homographies.append(np.eye(3))
                else:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x:x.distance)[:50]
                    # get points
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                    if len(src_pts) < 4:
                        homographies.append(np.eye(3))
                    else:
                        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                        if H is None:
                            H = np.eye(3)
                        homographies.append(H)
        else:
            # no registration
            homographies.append(np.eye(3))

    # Now do a second pass, applying the transforms
    # so we run the model in a stable coordinate system
    # We'll store the union in the reference frame
    if reference_gray is not None:
        # prepare an accumulator in reference size
        stable_accumulator = np.zeros((H_ref, W_ref), dtype=np.uint8)
    else:
        stable_accumulator = None

    # re-run the frames in the same order
    # (We do the model inference once, but for clarity we do it again, or you can store it in memory.)
    i_homo = 0
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(args.frames_dir, frame_file)
        original_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if original_img is None:
            i_homo += 1
            continue

        current_H = homographies[i_homo]
        i_homo += 1

        # 1) Run model as usual
        H_orig, W_orig = original_img.shape[:2]
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        aug = transforms(image=img_rgb)
        input_tensor = aug["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_upsampled = cv2.resize(pred.astype(np.uint8),(W_orig,H_orig), interpolation=cv2.INTER_NEAREST)

        # 2) Extract forceps + rhexis
        forceps_mask = (pred_upsampled == args.forceps_class_id).astype(np.uint8)
        rhexis_mask = (pred_upsampled == args.rhexis_class_id).astype(np.uint8)

        # 3) Find instrument tip bounding rect => get mm/pixel
        cnts_forceps, _ = cv2.findContours(forceps_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tip_center = None
        mm_per_pixel = 0.0
        if len(cnts_forceps)>0:
            biggest = max(cnts_forceps, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(biggest)
            tip_center = (x + w/2.0, y + h/2.0)
            if max(w,h)>0:
                mm_per_pixel = args.known_forceps_size_mm / float(max(w,h))

        # 4) Check if tip moves too far => skip frame
        if last_tip_center is not None and tip_center is not None:
            dx = tip_center[0] - last_tip_center[0]
            dy = tip_center[1] - last_tip_center[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > args.skip_threshold:
                # suspicious jump => skip
                continue

        if tip_center is not None:
            last_tip_center = tip_center
        if mm_per_pixel>0:
            scales.append(mm_per_pixel)

        # 5) "Register" the rhexis_mask into reference frame if requested
        if stable_accumulator is not None:
            # warp rhexis_mask using current_H
            # convert to float mask for warp
            float_mask = rhexis_mask.astype(np.float32)
            stable_mask = cv2.warpPerspective(float_mask, current_H, (W_ref, H_ref))
            stable_mask_bin = (stable_mask>0.5).astype(np.uint8)
            # union or intersection or last
            if args.union_strategy=="union":
                stable_accumulator = cv2.bitwise_or(stable_accumulator, stable_mask_bin)
            elif args.union_strategy=="intersection":
                stable_accumulator = cv2.bitwise_and(stable_accumulator, stable_mask_bin)
            else:
                # last
                stable_accumulator = stable_mask_bin.copy()
        else:
            # no registration => normal accumulation
            if accumulated_rhexis is None:
                accumulated_rhexis = rhexis_mask.copy()
            else:
                if args.union_strategy == "union":
                    accumulated_rhexis = cv2.bitwise_or(accumulated_rhexis, rhexis_mask)
                elif args.union_strategy == "intersection":
                    accumulated_rhexis = cv2.bitwise_and(accumulated_rhexis, rhexis_mask)
                else:
                    accumulated_rhexis = rhexis_mask.copy()

        # optionally save overlay
        if args.save_overlay:
            overlay = original_img.copy()
            cv2.drawContours(overlay, cnts_forceps, -1, (0,255,0), 2)
            conts_r, _ = cv2.findContours(rhexis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, conts_r, -1, (0,0,255), 2)
            out_name = os.path.splitext(frame_file)[0] + "_overlay.jpg"
            cv2.imwrite(os.path.join(args.output_dir,out_name), overlay)

    # end loop over frames

    # pick final union mask
    if stable_accumulator is not None:
        final_mask = stable_accumulator
        H_final, W_final = final_mask.shape
    else:
        final_mask = accumulated_rhexis
        H_final, W_final = final_mask.shape

    if final_mask is None:
        print("No frames processed or no rhexis found.")
        return

    # compute average scale
    if len(scales)>0:
        final_scale = sum(scales)/len(scales)
    else:
        final_scale = 0.0

    # measure final
    diameter_px, circ, off = measure_capsulorhexis_region(final_mask)
    diameter_mm = diameter_px * final_scale

    # final morphological check => e.g. if circ < some threshold, suspect it's not truly a rhexis
    if circ < args.circularity_filter:
        print(f"WARNING: Final shape has circularity={circ:.2f}, < {args.circularity_filter:.2f}, might be invalid.")

    # Optionally produce final overlay
    if args.save_overlay:
        final_overlay = np.zeros((H_final, W_final,3), dtype=np.uint8)
        conts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_overlay, conts, -1, (0,0,255), -1)
        cv2.imwrite(os.path.join(args.output_dir,"accumulated_rhexis_overlay.jpg"), final_overlay)

    # save final measurement
    import pandas as pd
    df = pd.DataFrame([{
        "diameter_mm": diameter_mm,
        "circularity": circ,
        "center_offset_px": off
    }])
    csv_path = os.path.join(args.output_dir, "capsulorhexis_final_measurement.csv")
    df.to_csv(csv_path, index=False)
    print("Final Capsulorhexis =>", df.to_dict(orient='records'))

if __name__ == "__main__":
    main()