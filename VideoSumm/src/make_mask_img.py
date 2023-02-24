import cv2
import torch
import numpy as np


def make_mask_img(outputs, input_data_img):
    proba_threshold = 0.5
    mask_img = []

    for i in range(len(outputs)):
        for j in range(len(outputs[i]["labels"])):
            outputs[i]["masks"][j] = np.where(outputs[i]["masks"][j] < proba_threshold, 0, 255)
            dst = input_data_img[i].copy()
            dst = cv2.bitwise_and(dst, dst, mask=outputs[i]["masks"][j].astype("uint8"))
            mask_img.append(dst)

    result = [(a, b) for a, b in zip(outputs, mask_img)]
    result.sort(key=lambda x: (x[0]["labels"], ((x[0]["boxes"][0][2] - x[0]["boxes"][0][0]) / (x[0]["boxes"][0][3] - x[0]["boxes"][0][1]))))

    mask_img = [a[1] for a in result]

    return mask_img

    