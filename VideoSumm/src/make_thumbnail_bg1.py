import cv2
import numpy as np


def make_thumbnail_bg1(dst1, bg_image, bg_c="sky", text_f="base", text_c="white",text="VLOG", font_scale=2, font_thickness=2):
    tmp_height, tmp_width = dst1.shape[0], dst1.shape[1]
    bg_color = {"sky": cv2.merge([np.full((tmp_height, tmp_width), 255, np.uint8), np.full((tmp_height, tmp_width), 204, np.uint8), np.full((tmp_height, tmp_width), 153, np.uint8)]),
        "pink": cv2.merge([np.full((tmp_height, tmp_width), 255, np.uint8), np.full((tmp_height, tmp_width), 51, np.uint8), np.full((tmp_height, tmp_width), 255, np.uint8)]),
        "red": cv2.merge([np.full((tmp_height, tmp_width), 0, np.uint8), np.full((tmp_height, tmp_width), 0, np.uint8), np.full((tmp_height, tmp_width), 255, np.uint8)]),
        }
    text_color = {"black": (0, 0, 0), "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255), "yellow": (204, 255, 255), "orange": (51, 153, 255)}
    text_font = {"SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX, "TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX, "ITALIC": cv2.FONT_ITALIC, "base": cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC}
    
    if bg_image is not None:
        dst2 = bg_image
    else:
        dst2 = bg_color[bg_c]

    # text_size = cv2.getTextSize(text, text_font[text_f], font_scale, font_thickness)[0]
    # text_x = int((dst2.shape[1] - text_size[0]) / 2)
    # text_y = int((dst2.shape[0] - text_size[1]) / 4 * 3)
    dst1_gray = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(dst1_gray, 0.1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    masked_fg = cv2.bitwise_and(dst1, dst1, mask=mask)
    masked_bg = cv2.bitwise_and(dst2, dst2, mask=mask_inv)
    dst = masked_fg + masked_bg

    # cv2.putText(dst, text, (text_x, text_y), text_font[text_f], font_scale, text_color[text_c], font_thickness)
    dst = cv2.resize(dst, (800, 450))
    return dst


