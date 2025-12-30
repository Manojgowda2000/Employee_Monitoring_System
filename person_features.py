import cv2
import numpy as np

def upper_body_crop(frame, x1, y1, x2, y2, frac=0.45):
    """
    Return the crop corresponding to the upper body.
    frac: fraction of bbox height to take from the top (default 0.45 -> top 45%)
    """
    h = max(1, y2 - y1)
    top_h = int(h * frac)
    uy1 = max(0, int(y1))
    uy2 = max(0, int(y1 + top_h))
    ux1 = max(0, int(x1))
    ux2 = max(0, int(x2))
    if uy2 <= uy1 or ux2 <= ux1:
        return None
    return frame[uy1:uy2, ux1:ux2]

def compute_hsv_hist(image, h_bins=32, s_bins=32):
    """
    Compute a 2D HSV histogram for H and S channels, normalize and return flattened array (float32).
    """
    if image is None or image.size == 0:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)

def hist_to_list(hist):
    """Convert numpy hist to Python list for JSON serialization."""
    if hist is None:
        return []
    return hist.tolist()

def list_to_hist(lst):
    """Convert python list back to numpy float32 array."""
    if not lst:
        return None
    return np.array(lst, dtype=np.float32)
