import cv2
import numpy as np


def imread(img_bytes):
    jpg_as_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
