import cv2
import numpy as np

def track_instrument_and_detect_capsule(frames, seg_model, device, threshold=0.5):
    """
    1) Segment each frame to find relevant instrument.
    2) Track the tip of the instrument in 2D coordinates.
    3) Once a 'circular path' is detected, generate a binary mask of that path.

    frames: list of np.array (H,W,3) in RGB
    seg_model: trained Mask R-CNN model
    device: 'cpu' or 'cuda'
    threshold: detection threshold for mask confidence

    Returns: final_mask (H, W) in {0,1} if circle is detected, else None
    """

    # We'll store all tip points in a list
    tip_points = []
    height, width, _ = frames[0].shape
    final_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        # Convert to appropriate input for seg_model
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_tensor = _preprocess_frame(frame, device)

        with torch.no_grad():
            predictions = seg_model([frame_tensor])[0]

        # predictions["masks"] -> list of (1, H, W)
        # predictions["labels"] -> predicted class IDs
        # predictions["scores"] -> confidence
        # We want the relevant instrument: 7, 11, or 12 (capsulorhexis instruments)
        # We'll pick the highest score among relevant labels
        chosen_idx = None
        max_score = 0.0
        for i, lbl in enumerate(predictions["labels"]):
            lbl_int = lbl.item()
            scr = predictions["scores"][i].item()
            if scr > threshold and lbl_int in [7, 11, 12]:
                if scr > max_score:
                    max_score = scr
                    chosen_idx = i

        if chosen_idx is not None:
            # Get mask
            mask = predictions["masks"][chosen_idx][0].cpu().numpy()
            # Convert to binary
            mask_bin = (mask > 0.5).astype(np.uint8)

            # Find the tip (the top-most or front-most pixel?)
            # Let's do a simple approach: find the contour of mask, pick an extreme point
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                biggest_contour = max(contours, key=cv2.contourArea)
                # Suppose the tip is the point on the contour with the smallest y (i.e. top)
                topmost = min(biggest_contour, key=lambda c: c[0][1])
                tip_points.append(topmost[0])  # (x, y)

            # Optional: draw the tip on the final_mask to visualize
            # but not necessary

        # Now check if these tip points form a circle
        # Heuristics: if we have enough points, and the distribution covers ~360 deg around a centroid
        if len(tip_points) > 30:
            circ_mask = _check_and_create_circle(tip_points, (height, width))
            if circ_mask is not None:
                final_mask = circ_mask
                return final_mask

    # If we never detect a full circle
    return None

def _preprocess_frame(frame_rgb, device):
    import torch
    frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float() / 255.
    frame_tensor = frame_tensor.unsqueeze(0).to(device)
    return frame_tensor[0]

def _check_and_create_circle(tip_points, shape_hw):
    """
    Heuristic:
    1) Find centroid of tip_points.
    2) Convert each point to polar angle around centroid.
    3) Check if angles fill ~360 degrees.
    If yes, create a binary mask of that circle region.
    """
    if len(tip_points) < 10:
        return None
    pts_arr = np.array(tip_points, dtype=np.float32)
    cx = np.mean(pts_arr[:,0])
    cy = np.mean(pts_arr[:,1])

    angles = []
    for (x, y) in pts_arr:
        dx = x - cx
        dy = y - cy
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    angles = np.sort(np.array(angles))

    # Check coverage of angles
    # A naive approach: measure the difference between min and max angle
    angle_range = angles[-1] - angles[0]
    if angle_range < np.deg2rad(300):
        # Maybe not enough coverage
        return None

    # If the coverage is large, we guess it's a near-complete circle
    # We can estimate radius from the average distance
    dists = []
    for (x,y) in pts_arr:
        d = np.sqrt((x-cx)**2 + (y-cy)**2)
        dists.append(d)
    radius = np.mean(dists)

    # Create a binary mask
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    # Approx circle
    cv2.circle(mask, center=(int(cx), int(cy)), radius=int(radius), color=1, thickness=-1)
    return mask
