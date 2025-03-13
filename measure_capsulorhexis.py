import numpy as np
import cv2

def measure_capsulorhexis_region(rhexis_mask):
    """
    Compute diameter, circularity, and center offset for a segmented capsulorhexis region.
    Args:
        rhexis_mask (np.ndarray): Binary mask (H,W) where 1 indicates rhexis region.
    Returns:
        diameter_px (float)
        circularity (float)
        center_offset (float)  -- offset in px from the image center for demonstration
    """
    # Find contours
    contours, _ = cv2.findContours(rhexis_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0, 0.0, 0.0  # or some default

    # Assume largest contour is the rhexis
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Diameter: approximate using bounding circle
    (x_c, y_c), radius = cv2.minEnclosingCircle(largest_contour)
    diameter_px = 2 * radius

    # Circularity = 4π * (area / perimeter^2) if perimeter != 0
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))

    # Center offset from image center
    H, W = rhexis_mask.shape
    img_center_x = W / 2.0
    img_center_y = H / 2.0
    dx = x_c - img_center_x
    dy = y_c - img_center_y
    center_offset = np.sqrt(dx*dx + dy*dy)

    return diameter_px, circularity, center_offset
