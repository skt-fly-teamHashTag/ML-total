import cv2
import torch
import numpy as np


def mask_preprocess(outputs, input_data):
    tmp_dic = {}
    prior_label = [1, 18, 17]
    del_list = []
    label_threshold = 0.9
    img_case = -1
    width_height_ratio = 2.5
    tmp_height, tmp_width = input_data[0][0].shape[:2]

    for i in prior_label:
        tmp_dic[i] = []

    for i in range(len(outputs)):
        outputs[i]["labels"] = outputs[i]["labels"].detach().numpy()
        outputs[i]["scores"] = outputs[i]["scores"].detach().numpy()
        outputs[i]["boxes"] = outputs[i]["boxes"].detach().numpy()
        outputs[i]["masks"] = torch.squeeze(outputs[i]["masks"], 1)
        outputs[i]["masks"] = outputs[i]["masks"].detach().numpy()
        del_idx = []

        for j in range(1, len(outputs[i]["labels"])):
            if outputs[i]["labels"][j] in outputs[i]["labels"][:j]:
                del_idx.append(j)

        outputs[i]["labels"] = np.delete(outputs[i]["labels"], del_idx)
        outputs[i]["scores"] = np.delete(outputs[i]["scores"], del_idx)
        outputs[i]["boxes"] = np.delete(outputs[i]["boxes"], del_idx, axis=0)
        outputs[i]["masks"] = outputs[i]["masks"].reshape(outputs[i]["masks"].shape[0], -1)
        outputs[i]["masks"] = np.delete(outputs[i]["masks"], del_idx, axis=0)

    for i in range(len(outputs)):
        del_idx = []
        for j in range(len(outputs[i]["labels"])):
            if outputs[i]["labels"][j] in tmp_dic.keys():
                if (outputs[i]["boxes"][j][3] - outputs[i]["boxes"][j][1]) / (outputs[i]["boxes"][j][2] - outputs[i]["boxes"][j][0]) > width_height_ratio:
                    del_idx.append(j)
            else:
                del_idx.append(j)

        outputs[i]["labels"] = np.delete(outputs[i]["labels"], del_idx)
        outputs[i]["scores"] = np.delete(outputs[i]["scores"], del_idx)
        outputs[i]["boxes"] = np.delete(outputs[i]["boxes"], del_idx, axis=0)
        outputs[i]["masks"] = np.delete(outputs[i]["masks"], del_idx, axis=0)
    
    for i in range(len(outputs)):
        for j in range(len(outputs[i]["labels"])):
            if outputs[i]["labels"][j] in prior_label:
                tmp_dic[outputs[i]["labels"][j]].append(outputs[i]["scores"][j])
                
    for i in list(tmp_dic.keys()):
        del_list = []
        for j in tmp_dic[i]:
            if j < label_threshold:
                del_list.append(j)

        for d in del_list:
            tmp_dic[i].remove(d)

        tmp_dic[i].sort(reverse=True)

    if len(tmp_dic[1]) != 0:   # 사람 O
        if len(tmp_dic[18]) == 0 and len(tmp_dic[17]) == 0:   # 사람 O, 동물 X
            img_case = 1
            while len(tmp_dic[1]) > 3:
                tmp_dic[1].pop()
        else:   # 사람 O, 동물 O
            img_case = 2
            if len(tmp_dic[18]) != 0:
                tmp_dic[18] = [max(tmp_dic[18])]
            if len(tmp_dic[17]) != 0:
                tmp_dic[17] = [max(tmp_dic[17])]
            while len(tmp_dic[1]) > 2:
                tmp_dic[1].pop()

    else:   # 사람 X
        if len(tmp_dic[18]) == 0 and len(tmp_dic[17]) == 0:   # 사람 X, 동물 X
            img_case = 4
        else:   # 사람 X, 동물 O
            img_case = 3
            while len(tmp_dic[18]) > 2:
                tmp_dic[18].pop()
            while len(tmp_dic[17]) > 2:
                tmp_dic[17].pop()
            while len(tmp_dic[18] + tmp_dic[17]) > 2:
                min_tmp = min(tmp_dic[18] + tmp_dic[17])
                if min_tmp in tmp_dic[17]:
                    tmp_dic[17].remove(min_tmp)
                else:
                    tmp_dic[18].remove(min_tmp)

    for i in range(len(outputs)):
        del_idx = []
        for j in range(len(outputs[i]["labels"])):
            if outputs[i]["labels"][j] in tmp_dic.keys():
                if outputs[i]["scores"][j] not in tmp_dic[outputs[i]["labels"][j]]:
                    del_idx.append(j)
            else:
                del_idx.append(j)

        outputs[i]["labels"] = np.delete(outputs[i]["labels"], del_idx)
        outputs[i]["scores"] = np.delete(outputs[i]["scores"], del_idx)
        outputs[i]["boxes"] = np.delete(outputs[i]["boxes"], del_idx, axis=0)
        outputs[i]["masks"] = np.delete(outputs[i]["masks"], del_idx, axis=0)
        outputs[i]["masks"] = outputs[i]["masks"].reshape(outputs[i]["masks"].shape[0], tmp_height, tmp_width)

    tmp_idx = []
    tmp_outputs = []
    tmp_input_data = []
    for i in range(len(outputs)):
        if len(outputs[i]["labels"]) != 0:
            tmp_idx.append(i)

    for i in tmp_idx:
        tmp_outputs.append(outputs[i])
        tmp_input_data.append(input_data[i][0].copy())

    # outputs = tmp_outputs
    # input_data_img = tmp_input_data   
    return img_case, tmp_outputs, tmp_input_data   