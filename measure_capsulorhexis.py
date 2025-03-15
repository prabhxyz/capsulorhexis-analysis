import numpy as np
import cv2

def measure_capsulorhexis_region(rhexis_mask):
    # Largest external contour => diameter, circularity, center offset
    contours, _ = cv2.findContours(rhexis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0, 0.0, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    diameter_px = 2.0 * radius
    circularity = 0.0
    if perimeter > 0:
        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))

    H, W = rhexis_mask.shape[:2]
    dx = cx - (W / 2.0)
    dy = cy - (H / 2.0)
    center_offset = np.sqrt(dx*dx + dy*dy)
    return diameter_px, circularity, center_offset
