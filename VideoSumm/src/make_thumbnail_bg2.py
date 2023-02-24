import cv2
import torch
import numpy as np


def make_thumbnail_bg2(background_img, background_output, text_f="base", text_c="white",text="VLOG", font_scale=2, font_thickness=2):
    tmp_height, tmp_width = background_img[0].shape[:2]
    text_font = {"SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX, "TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX, "ITALIC": cv2.FONT_ITALIC, "base": cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC}
    text_color = {"black": (0, 0, 0), "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255)}
    dst = np.zeros((tmp_height, tmp_width, 3), np.uint8)
    tmp_img_width = int(tmp_width // 3)
    
    for i in range(len(background_output)):
        if len(background_output[i]["labels"]) != 0:
            background_output[i]["labels"] = [background_output[i]["labels"].detach().numpy()[0]]
            background_output[i]["scores"] = background_output[i]["scores"].detach().numpy()[0]
            background_output[i]["boxes"] = background_output[i]["boxes"].detach().numpy()[0].astype("uint16")
            background_output[i]["masks"] = torch.squeeze(background_output[i]["masks"], 1)
            background_output[i]["masks"] = background_output[i]["masks"].detach().numpy()[0]
        else:
            background_output[i]["labels"] = background_output[i]["labels"].detach().numpy()

    for i in range(len(background_output)):
        if len(background_output[i]["labels"]) != 0:
            pt = int((background_output[i]["boxes"][2] - background_output[i]["boxes"][0]) // 2 + background_output[i]["boxes"][0])
            
            if pt - tmp_img_width // 2 < 0:
                dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = background_img[i][:, :tmp_img_width]
            elif pt + tmp_img_width // 2 > tmp_width:
                dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = background_img[i][:,tmp_img_width * -1:]
            else:
                if (pt + tmp_img_width // 2) - (pt - tmp_img_width // 2) > tmp_img_width:
                    k = ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2)) - tmp_img_width
                else:
                    k = tmp_img_width - ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2))
                dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = background_img[i][:, pt - tmp_img_width // 2:pt + tmp_img_width // 2 + k]
        
        else:
            pt = int(background_img[i].shape[1] // 2)
            if (pt + tmp_img_width // 2) - (pt - tmp_img_width // 2) > tmp_img_width:
                k = ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2)) - tmp_img_width
            else:
                k = tmp_img_width - ((pt + tmp_img_width // 2) - (pt - tmp_img_width // 2))
            dst[:, i * tmp_img_width:(i+1) * tmp_img_width] = background_img[i][:, pt - tmp_img_width // 2:pt + tmp_img_width // 2 + k]

    # text_size = cv2.getTextSize(text, text_font[text_f], font_scale, font_thickness)[0]
    # text_x = int((bg.shape[1] - text_size[0]) / 2)
    # text_y = int((bg.shape[0] - text_size[1]) / 4 * 3)

    # cv2.putText(bg, text, (text_x, text_y), text_font[text_f], font_scale, text_color[text_c], font_thickness)
    dst = cv2.resize(dst, (800, 450))
    dst = cv2.rectangle(dst, (10, 10), (dst.shape[1] - 10, dst.shape[0] - 10), (255, 255, 255), 2)
    dst = cv2.rectangle(dst, (20, 20), (dst.shape[1] - 20, dst.shape[0] - 20), (255, 255, 255), 2)

    return dst


