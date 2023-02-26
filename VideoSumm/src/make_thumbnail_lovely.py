import cv2
import torch
import numpy as np
import skimage.exposure
from font_bg import love_make_image


def make_thumbnail_lovely(outputs, input_data, message):
    max_score = -1
    seg_img_idx = -1
    seg_img_person_idx = -1
    back_img_idx = []
    back_img_list = []
    x = 0.2
    message = message

    for i in range(len(outputs)):
        outputs[i]["labels"] = outputs[i]["labels"].detach().numpy()
        outputs[i]["scores"] = outputs[i]["scores"].detach().numpy()
        outputs[i]["boxes"] = outputs[i]["boxes"].detach().numpy()
        outputs[i]["masks"] = torch.squeeze(outputs[i]["masks"], 1)
        outputs[i]["masks"] = outputs[i]["masks"].detach().numpy()

        if 1 in outputs[i]["labels"]:
            back_img_idx.append(i)
            k = np.where(outputs[i]["labels"] == 1)[0][0]
            if outputs[i]["scores"][k] > max_score:
                max_score = outputs[i]["scores"][k]
                seg_img_idx = i
                seg_img_person_idx = k

    back_img_idx.remove(seg_img_idx)

    while len(back_img_idx) < 4:
        back_img_idx.append(int(len(input_data) * x)[0])
        x += 0.2

    back_img_list.append(input_data[back_img_idx[0]][0])
    back_img_list.append(input_data[back_img_idx[-1]][0])   
    back_img_list.append(input_data[int(len(back_img_idx) * 0.3)][0])
    back_img_list.append(input_data[int(len(back_img_idx) * 0.6)][0])

    for i in range(len(back_img_list)):
        back_img_list[i] = cv2.resize(back_img_list[i], (400, 225))

    dst = np.zeros((450, 800, 3), np.uint8)
    dst[:225, :400] = back_img_list[0]
    dst[:225, 400:] = back_img_list[1]
    dst[225:, :400] = back_img_list[2]
    dst[225:, 400:] = back_img_list[3]

    outputs[seg_img_idx]["masks"][seg_img_person_idx] = np.where(outputs[seg_img_idx]["masks"][seg_img_person_idx] < 0.5, 0, 255)
    seg_img = cv2.bitwise_and(input_data[seg_img_idx][0], input_data[seg_img_idx][0], mask=outputs[seg_img_idx]["masks"][seg_img_person_idx].astype("uint8"))
    seg_img = cv2.resize(seg_img, (800, 450))

    tmp = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)[1]

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
    (x, y, w, h, area) = stats[1]
    mat = np.float32([[1, 0, -x + 20], [0, 1, -y + 20]])
    seg_img = cv2.warpAffine(seg_img, mat, (0, 0))

    k = (450) / (h * 2)
    seg_img = cv2.resize(seg_img, (None, None), fx=k, fy=k, interpolation=cv2.INTER_AREA)
    bg = np.zeros((450, 800, 3), np.uint8)
    bg[:seg_img.shape[0], :seg_img.shape[1]] = seg_img[:seg_img.shape[0], :seg_img.shape[1]]

    tmp = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

    a, b = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(bg, a, -1, (255, 255, 255), 3)

    tmp = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(tmp, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
    result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
    result = result.astype(np.uint8)

    bg = cv2.bitwise_and(bg, bg, mask=result)

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
    (x, y, w, h, area) = stats[1]
    mat = np.float32([[1, 0, 400 - w // 2 - 20], [0, 1, 225 - h // 2 - 20]])
    bg = cv2.warpAffine(bg, mat, (0, 0))

    tmp = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY_INV)[1] 

    dst = cv2.bitwise_and(dst, dst, mask=tmp)
    dst = dst + bg

    dst = cv2.resize(dst, (780, 430))

    pink_bg = cv2.merge([np.full((450, 800), 229, np.uint8), np.full((450, 800), 204, np.uint8), np.full((450, 800), 255, np.uint8)])
    pink_bg[10:440, 10:790] = 0
    pink_bg[10:440, 10:790] = dst

    pink_bg = love_make_image(message, pink_bg)

    return pink_bg

    