import cv2
import torch
import random
from thumbnail import Thumbnail
from mask_preprocess import mask_preprocess
from make_mask_img import make_mask_img
from make_thumbnail_fg import make_thumbnail_fg
from make_thumbnail_bg1 import make_thumbnail_bg1
from make_thumbnail_bg2 import make_thumbnail_bg2
from make_thumbnail_daily import make_thumbnail_daily
from make_thumbnail_lovely import make_thumbnail_lovely
from make_thumbnail_modern import make_thumbnail_modern


def make_thumbnail(thumb_numpy, nickname, category_list):
    a = Thumbnail(thumb_numpy, step=5)
    vlog_message = nickname
    category = -1
    # category_dic = {0: "가족", 1: "스터디", 2: "뷰티", 3: "반려동물", 4: "운동/스포츠", 5: "음식", 6: "여행", 7: "연애/결혼", 8: "문화생활", 9: "직장인"}

    if random.choice([0, 1]) == 0:
        category = category_list[0]
    else:
        category = category_list[1]
    
    a.forward()

    if category in [2, 7]:
        dst = make_thumbnail_lovely(outputs=a.outputs, input_data=a.input_data, message=vlog_message)
        return dst

    img_case, outputs, input_data = mask_preprocess(outputs=a.outputs, input_data=a.input_data)

    if img_case == 4:
        background_img_list = []

        for i in range(len(a.background_img)):
            background_img_list.append(a.tt(a.background_img[i]))

        torch_bg_list = [a.transform(tmp) for tmp in background_img_list]

        with torch.no_grad():
            background_output = a.model(torch_bg_list)
        dst = make_thumbnail_bg2(background_img=a.background_img, background_output=background_output, text_f="base", text_c="white", text=vlog_message, font_scale=2, font_thickness=2)

    else:
        if category in [1, 4, 9]:
            dst = make_thumbnail_modern(input_data=input_data, outputs=outputs, message=vlog_message)
        else:
            mask_img = make_mask_img(outputs=outputs, input_data_img=input_data)
            dst = make_thumbnail_fg(img_case, mask_img)
            dst = make_thumbnail_bg1(dst1=dst, bg_image=a.background_img1, bg_c="sky", text_f="base", text_c="white",text=vlog_message, font_scale=2, font_thickness=2)
            dst = cv2.resize(dst, (780, 430))
            dst = make_thumbnail_daily(img=dst, message=vlog_message)
            
    dst = cv2.resize(dst, (800, 450))
    return dst

