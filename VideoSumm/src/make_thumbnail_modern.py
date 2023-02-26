import cv2
import numpy as np
from font_bg import partition_make_image


def make_thumbnail_modern(input_data, outputs, message):
    tmp_height, tmp_width = input_data[0].shape[:2]
    message = message
    dst = np.zeros((tmp_height, tmp_width, 3), np.uint8)
    tmp_img_width = int(tmp_width // 3)
    
    for i in range(len(outputs)):
        if len(outputs[i]["labels"]) != 0:
            pt = int((outputs[i]["boxes"][0][2] - outputs[i]["boxes"][0][0]) // 2 + outputs[i]["boxes"][0][0])
            
            if pt - tmp_img_width // 2 < 0:
                dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = input_data[i][:, :tmp_img_width]
            elif pt + tmp_img_width // 2 > tmp_width:
                dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = input_data[i][:,tmp_img_width * -1:]
            else:
                if (pt + tmp_img_width // 2) - (pt - tmp_img_width // 2) > tmp_img_width:
                    k = ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2)) - tmp_img_width
                else:
                    k = tmp_img_width - ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2))
                dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = input_data[i][:, pt - tmp_img_width // 2:pt + tmp_img_width // 2 + k]
        
        else:
            pt = int(input_data[i].shape[1] // 2)
            if (pt + tmp_img_width // 2) - (pt - tmp_img_width // 2) > tmp_img_width:
                k = ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2)) - tmp_img_width
            else:
                k = tmp_img_width - ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2))
            dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = input_data[i][:, pt - tmp_img_width // 2:pt + tmp_img_width // 2 + k]

    dst = cv2.resize(dst, (800, 450))
    dst = cv2.rectangle(dst, (10, 10), (dst.shape[1] - 10, dst.shape[0] - 10), (255, 255, 255), 2)
    dst = cv2.rectangle(dst, (20, 20), (dst.shape[1] - 20, dst.shape[0] - 20), (255, 255, 255), 2)

    dst = partition_make_image(message, dst)

    return dst

