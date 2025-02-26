import cv2
import numpy as np

def measure_rhexis(rhexis_mask, reference_instrument_size=1.5):
    """
    rhexis_mask: (H, W) binary mask of the capsulorhexis.
    reference_instrument_size: approximate mm for some known portion (e.g., typical forceps width).
        This can be used to approximate real-world measurements.

    Returns: dict with diameter_mm, circularity, center=(cx, cy)
    """
    # Find contour
    contours, _ = cv2.findContours(rhexis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return {
            "diameter_mm": 0,
            "circularity": 0,
            "center": (0,0)
        }
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return {
            "diameter_mm": 0,
            "circularity": 0,
            "center": (0,0)
        }
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # Get bounding box to approximate diameter
    x, y, w, h = cv2.boundingRect(c)
    avg_diam_pixels = (w + h) / 2.0
    # Convert to mm using reference_instrument_size somehow
    # For a real approach, we'd calibrate more carefully
    # We'll do a naive approach: assume reference_instrument_size mm = 10 pixels
    # => 1 pixel = reference_instrument_size / 10 mm
    # => diameter in mm = avg_diam_pixels * (reference_instrument_size/10)
    px_scale = reference_instrument_size / 10.0
    diameter_mm = avg_diam_pixels * px_scale

    # Compute circularity = 4*pi*Area / (Perimeter^2)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    return {
        "diameter_mm": diameter_mm,
        "circularity": circularity,
        "center": (cx, cy)
    }
