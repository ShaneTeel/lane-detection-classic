import numpy as np

THRESH_WHITE_LOWER = np.array([np.round(  0 / 2), np.round(0.75 * 255), np.round(0.00 * 255)], dtype=np.int32)
THRESH_WHITE_UPPER = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)], dtype=np.int32)

THRESH_YELLOW_LOWER = np.array([np.round(40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)], dtype=np.int32)
THRESH_YELLOW_UPPER = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)], dtype=np.int32)