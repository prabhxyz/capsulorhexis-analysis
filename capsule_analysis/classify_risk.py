def classify_rhexis(rhexis_info, center_threshold=1.0, ideal_diameter_range=(4.5, 5.5), circularity_threshold=0.8, eye_center=(256,256)):
    """
    Rule-based classification.

    rhexis_info: dict with diameter_mm, circularity, center=(cx,cy)
    center_threshold: offset in mm considered acceptable
    ideal_diameter_range: acceptable rhexis diameter range in mm
    circularity_threshold: lower bound for "good roundness"
    eye_center: approximate center of the pupil/visual axis in pixel coords
        (in real usage, we'd track pupil center from segmentation, etc.)

    Returns: (label, reasons)
        label in { "Ideal", "At-Risk" }
        reasons: list of strings explaining why
    """

    label = "Ideal"
    reasons = []

    # Check diameter
    diam = rhexis_info["diameter_mm"]
    if diam < ideal_diameter_range[0]:
        label = "At-Risk"
        reasons.append(f"Rhexis too small ({diam:.2f} mm).")
    elif diam > ideal_diameter_range[1]:
        label = "At-Risk"
        reasons.append(f"Rhexis too large ({diam:.2f} mm).")

    # Check circularity
    circ = rhexis_info["circularity"]
    if circ < circularity_threshold:
        label = "At-Risk"
        reasons.append(f"Rhexis not round enough (circularity={circ:.2f}).")

    # Check centering - naive approach: measure pixel distance from eye_center
    # We'll convert it to mm using the same scale assumption:
    px_scale = 1.5 / 10.0
    (cx, cy) = rhexis_info["center"]
    dx = (cx - eye_center[0])
    dy = (cy - eye_center[1])
    dist_px = (dx**2 + dy**2)**0.5
    dist_mm = dist_px * px_scale

    if dist_mm > center_threshold:
        label = "At-Risk"
        reasons.append(f"Rhexis off-center by {dist_mm:.2f} mm.")

    return label, reasons
