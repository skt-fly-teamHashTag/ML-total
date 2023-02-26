import cv2
import numpy as np
import skimage.exposure


def make_thumbnail_fg(img_case, mask_img):
    img_list = []
    mask_img_copy = mask_img
    tmp_height, tmp_width = mask_img[0].shape[0], mask_img[0].shape[1]
    tmp_img_size = tmp_height * tmp_width
    dst = np.zeros((tmp_height, tmp_width, 3), np.uint8)

    for i in range(len(mask_img)):
        tmp = cv2.cvtColor(mask_img[i], cv2.COLOR_BGR2GRAY)
        _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)

        for j in range(1, cnt):
            (x, y, w, h, area) = stats[j]
            if area / tmp_img_size < 1 / 40:
                tmp[y:y+h, x:x+w] = 0
            
        a, b = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(a) != 0:
            tmp = cv2.bitwise_and(mask_img_copy[i], mask_img_copy[i], mask=tmp)
            img_list.append(tmp)
    
    if img_case == 1:
        for i in range(len(img_list)):
            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
            (x, y, w, h, area) = stats[1]

            m = np.float32([[1, 0, -x], [0, 1, -y]])
            img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))

            if i == 0:
                k = (tmp_height * 9) / (h * 10)
            else:
                k = (tmp_height * 9) / (h * 10)

            img_list[i] = cv2.resize(img_list[i], (None, None), fx=k, fy=k, interpolation=cv2.INTER_AREA)

            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            a, b = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(img_list[i], a, -1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawContours(img_list[i], a, -1, (255, 255, 255), 3)

            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            blur = cv2.GaussianBlur(tmp, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
            result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
            result = result.astype(np.uint8)

            img_list[i] = cv2.bitwise_and(img_list[i], img_list[i], mask=result)

    elif img_case == 2:
        for i in range(len(img_list)):
            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
            (x, y, w, h, area) = stats[1]

            m = np.float32([[1, 0, -x], [0, 1, -y]])
            img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))

            if i == 0:
                k = (tmp_height * 9) / (h * 10)
            elif i == 1:
                k = (tmp_height * 9) / (h * 10)
            else:
                k = (tmp_height * 3) / (h * 10)
            
            img_list[i] = cv2.resize(img_list[i], (None, None), fx=k, fy=k, interpolation=cv2.INTER_AREA)

            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            a, b = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(img_list[i], a, -1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawContours(img_list[i], a, -1, (255, 255, 255), 3)

            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            blur = cv2.GaussianBlur(tmp, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
            result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
            result = result.astype(np.uint8)

            img_list[i] = cv2.bitwise_and(img_list[i], img_list[i], mask=result)

    elif img_case == 3:
        for i in range(len(img_list)):
            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
            (x, y, w, h, area) = stats[1]

            m = np.float32([[1, 0, -x], [0, 1, -y]])
            img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))

            if i == 0:
                k = (tmp_height * 9) / (h * 10)
            else:
                k = (tmp_height * 7) / (h * 10)

            img_list[i] = cv2.resize(img_list[i], (None, None), fx=k, fy=k, interpolation=cv2.INTER_AREA)

            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            a, b = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(img_list[i], a, -1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawContours(img_list[i], a, -1, (255, 255, 255), 3)

            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

            blur = cv2.GaussianBlur(tmp, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
            result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
            result = result.astype(np.uint8)

            img_list[i] = cv2.bitwise_and(img_list[i], img_list[i], mask=result)

            
    for i in range(len(img_list)):
        bg = np.zeros((tmp_height, tmp_width, 3), dtype=np.uint8)

        if img_list[i].shape[0] > bg.shape[0]:
            bg[:bg.shape[0], :bg.shape[1]] = img_list[i][:bg.shape[0], :bg.shape[1]]
        else:
            bg[:img_list[i].shape[0], :img_list[i].shape[1]] = img_list[i]
        img_list[i] = bg


    if img_case == 1:
        for i in range(len(img_list)):
            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 1, 255, cv2.THRESH_BINARY)

            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
            centroids = centroids.astype(np.int64)
            (x, y, w, h, area) = stats[1]

            if i == 0:
                shift_x = (tmp_width - w) // 2
                shift_y = tmp_height - h
            elif i == 1:
                shift_x = tmp_width - w
                shift_y = tmp_height - h
            elif i == 2:
                shift_x = 0
                shift_y = tmp_height - h
            m = np.float32([[1, 0, int(shift_x)], [0, 1, int(shift_y)]])
            img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))
    
    elif img_case == 2:
        for i in range(len(img_list)):
            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 1, 255, cv2.THRESH_BINARY)

            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
            centroids = centroids.astype(np.int64)
            (x, y, w, h, area) = stats[1]

            if i == 0:
                shift_x = 0
                shift_y = tmp_height - h
            elif i == 1:
                shift_x = tmp_width - w
                shift_y = tmp_height - h
            elif i == 2:
                shift_x = tmp_width / 2 - w
                shift_y = 20
            else:
                shift_x = tmp_width - w - 20
                shift_y = 20
            m = np.float32([[1, 0, int(shift_x)], [0, 1, int(shift_y)]])
            img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))
    
    elif img_case == 3:
        for i in range(len(img_list)):
            tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            _, tmp = cv2.threshold(tmp, 1, 255, cv2.THRESH_BINARY)

            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
            centroids = centroids.astype(np.int64)
            (x, y, w, h, area) = stats[1]

            if i == 0:
                shift_x = 0
                shift_y = tmp_height - h
            else:
                shift_x = tmp_width - w - 10
                shift_y = tmp_height - h
                
            m = np.float32([[1, 0, int(shift_x)], [0, 1, int(shift_y)]])
            img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))


    for i in img_list[::-1]:
        gray_tmp = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_tmp, 0.1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        masked_fg = cv2.bitwise_and(i, i, mask=mask)
        masked_bg = cv2.bitwise_and(dst, dst, mask=mask_inv)

        dst = masked_fg + masked_bg
    
    return dst      


