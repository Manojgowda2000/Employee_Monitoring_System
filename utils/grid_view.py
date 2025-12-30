import cv2
import math
import numpy as np

def compute_grid(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def build_grid(frames, resize_w=900, resize_h=600):
    n = len(frames)
    rows, cols = compute_grid(n)

    resized = [cv2.resize(f, (resize_w, resize_h)) for f in frames]

    blank = np.zeros_like(resized[0])
    while len(resized) < rows * cols:
        resized.append(blank)

    grid = []
    for r in range(rows):
        row = np.hstack(resized[r*cols:(r+1)*cols])
        grid.append(row)

    return np.vstack(grid)