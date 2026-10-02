import cv2
import numpy as np

def clean_mask(mask, kernel_size=10, iterations=1):
    """
    Perform morphological operations to clean the mask.
    Args:
        mask: Binary mask (numpy array).
        kernel_size: Size of the kernel for morphological operations.
        iterations: Number of iterations for morphological operations.
    Returns:
        Cleaned binary mask.
    """
    # Convert mask to uint8 if it's boolean
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)

    # Apply opening to remove small noise and separate connected objects
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # Apply closing to fill small holes within objects
    kernel = np.ones((kernel_size//2, kernel_size//2), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return cleaned_mask


def mask_to_polygons(mask):
    """Convert binary mask to COCO polygon format using OpenCV."""
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.squeeze(1).astype(float)  # (N, 2)
        polygon = contour.flatten().tolist()        # [x1, y1, x2, y2, ...]
        polygons.append(polygon)
    return polygons