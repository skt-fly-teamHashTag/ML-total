import cv2
import numpy as np


def make_thumbnail_lovely(img):
    tmp_height, tmp_width = img.shape[:2]
    bg = cv2.merge([np.full((tmp_height + 20, tmp_width + 20), 119, np.uint8), np.full((tmp_height + 20, tmp_width + 20), 128, np.uint8), np.full((tmp_height + 20, tmp_width + 20), 255, np.uint8)])
    bg[10:tmp_height + 10, 10:tmp_width + 10] = 0
    bg[10:tmp_height + 10, 10:tmp_width + 10] = img
    return bg