import cv2
import numpy as np
from font_bg import poo_make_image


def make_thumbnail_daily(img, message):
    tmp_height, tmp_width = img.shape[:2]
    message = message
    bg = cv2.merge([np.full((450, 800), 0, np.uint8), np.full((450, 800), 212, np.uint8), np.full((450, 800), 255, np.uint8)])
    bg[10:440, 10:790] = 0
    bg[10:440, 10:790] = img

    print(bg.shape)
    dst = poo_make_image(message, img)
    print(dst.shape)
    
    return dst