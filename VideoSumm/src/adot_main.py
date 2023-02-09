import cv2
import numpy as np
import torch
import torchvision 
import matplotlib.pyplot as plt  

from helpers import init_helper, adot_vsumm_helper, adot_caption_helper, bbox_helper, video_helper
from modules.model_zoo import get_model
from dsnet_main import video_shot_main
from hashtag import TextRank
from thumbnail import make_mask, make_thumbnail_fg, make_thumbnail_bg

from torchvision.io.image import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from ofa_main import infer_main

from hashtag import TextRank
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def thumb_nail_main(input_data):

    original_height, original_width = input_data[0][0].shape[:2]
    original_img_size = original_height * original_width
    
    for i in input_data:
        if i[0].shape[1] > 1500:
            i[0] = cv2.resize(i[0], (None, None), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        elif i[0].shape[1] > 1000:
            i[0] = cv2.resize(i[0], (None, None), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

    background_img3 = input_data[0]   # 세그먼트할 대상이 없을 경우 사용할 배경(3개 이미지중 첫번째)
    background_img4 = input_data[len(input_data) // 2]   # 세그먼트할 대상이 없을 경우 사용할 배경(3개 이미지중 두번째)
    background_img5 = input_data[-1]   # 세그먼트할 대상이 없을 경우 사용할 배경(3개 이미지중 세번째)

    input_data.sort(reverse=True, key=lambda x:x[2])   # idx 1은 cps_score, idx 2는 frame_score, frame_score로 정렬

    background_img1 = input_data.pop()[0]   # 세그먼트할 대상이 있는 경우 사용할 배경
    background_img2 = input_data.pop(0)[0]   # 세그먼트할 대상이 없을 경우 사용할 배경(1개 이미지)
    tmp_height, tmp_width = input_data[0][0].shape[:2]
    tmp_img_size = tmp_height * tmp_width


    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transform = weights.transforms()

    model = maskrcnn_resnet50_fpn(weights=weights)
    model = model.eval()


    input_img_list = []
    tt = torchvision.transforms.ToTensor()

    for i in range(len(input_data)):
        input_img_list.append(tt(input_data[i][0]))


    torch_img_list = [transform(tmp) for tmp in input_img_list]

    outputs = model(torch_img_list)


    real_dic, img_case, mask_img = make_mask(outputs)
    dst1 = make_thumbnail_fg(mask_img)
    dst2 = make_thumbnail_bg(dst1, bg_image=True, bg_c="sky", text_f="TRIPLEX", text_c="white", text="QWE's VLOG", font_scale=2, font_thickness=2)

    return dst2

def translation_model(sentences):
    model_name = "QuoQA-NLP/KE-T5-En2Ko-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translated = model.generate(**tokenizer(sentences, return_tensors="pt", padding=True))
    ko_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return ko_sentences

def hashtag_main(sen):
    ko_sentences = translation_model(sen)
    textrank = TextRank(ko_sentences)
    hashtag = textrank.keywords()

    return hashtag 


if __name__ == '__main__':
    #####--- video summary & save ---######
    print('*** start video summary ***')
    thumb_input, caption_images = video_shot_main() # [[image, cps_score, frame_score], ...]
    # print(f'len(thumbnail_images): {len(thumb_input)}, len(caption_images): {len(caption_images)}')
    print('--- fisish viedo summary ---')
    # save_thumb = np.array(thumb_input)
    # np.save('../output/test/testdata3_output', save_thumb)
    # print("save thumb npy finished") 

    ######--- thumbnail generate ---###### 
    print('*** start make thumnail ***') 
    thumbnail_result = thumb_nail_main(thumb_input) 
    print('--- finish thumbnail ---') 

    ######--- captioning ---######
    print('*** start image captioning ***')
    sentences = infer_main(caption_images)
    print('--- finish image captioning ---')

    ######--- translation & extract hashtag ---######
    print('*** start extract hashtag ***')
    tag_out = hashtag_main(sentences)
    print('--- finish hashtag ---')

