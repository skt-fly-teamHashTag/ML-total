import cv2
import numpy as np
import torch
import torchvision 
import matplotlib.pyplot as plt  
from modules.model_zoo import get_model
from dsnet_main import video_shot_main, text_classification, makeSumm
from hashtag import TextRank
from qwer import qwe
import urllib.request
import os 

from torchvision.io.image import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# from ofa_main import infer_main
from expansion_main import caption_expansion

# from hashtag import TextRank
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from KeyBERThashtag import KeyBERTModel

## flask 
from flask import Flask, render_template, request, jsonify
app = Flask(__name__) #flask 앱 초기화

##multiprocessing 
from multiprocessing import Process, Pool 

def thumb_nail_main(input_data):
    '''
    input_data: [list] 
    '''
    thumbnail_output = qwe(input_data)
    
    # 썸네일 사진 저장 
    IMG_PATH = "thumbnail.jpg"
    cv2.imwrite(IMG_PATH, thumbnail_output)
    return IMG_PATH

    # return jsonify({'thumbnail path name':IMG_PATH})

def translation_model(sentences):
    model_name = "QuoQA-NLP/KE-T5-En2Ko-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translated = model.generate(**tokenizer(sentences, return_tensors="pt", padding=True))
    ko_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return ko_sentences

def hashtag_main(sen):

    ko_sentences = translation_model(sen)
    keybert = KeyBERTModel(ko_sentences)
    hashtag = keybert.keywords

    return hashtag 
    # return jsonify({'hashtag':hashtag})

# @app.route('/video_summary', methods=['POST'])
def test():
    return {
        'video_image': "output thumbnail image path", 
        'video_path': "output summ video path",
        'video_tag': ['#오늘', '#바다', '#가고싶다']
    }
    
def save_video(video_url_lst) :
    save_dir = "../origin_video"
    source_lst = [] 

    for i in range(len(video_url_lst)):
        video_url = video_url_lst[i]
        saveName = f"video_{i}.mp4"
        save_path = os.path.join(save_dir, saveName)
        urllib.request.urlretrieve(video_url, save_path)
        source_lst.append(save_path)

    return source_lst


@app.route('/video_summary', methods=['POST'])
def predict():

    data = request.get_json() 
    user_ID = data['user_id'] 
    video_src_lst = data['video_origin_src'] #list= ['video_scr1', 'video_src2', ..]
    nickname = data['nickname'] 
    category = data['category'] 

    source_lst = save_video(video_src_lst) #이 부분은 미리 저장해둬도 괜찮을 듯 
    print("video download successed from s3!!") 

    save_path = '../output/vlog.mp4'

    ##영상요약 
    # video preprocessing & STT & ObjectDetection 
    total_stt, ws_obj_lst, seq, model, cps, n_frames, nfps, picks, ws_cps = video_shot_main(source_lst) #thumb_input: type==list 
    # 구간의 음성 주제 분류 
    ws_score, hashtag = text_classification(category, total_stt, ws_obj_lst)
    # 가중치 & 요약 영상 만들기, return 썸네일 이미지 
    thumb_input = makeSumm(seq, model, cps, n_frames, nfps, picks, source_lst, save_path, ws_score, ws_cps)
    print(f'len(thumbnail_images): {len(thumb_input)}')
    print("video summary successed!!")

    ##썸네일 
    # thumb_input = np.load('../output/test/test7_class_thumb_9.npy', allow_pickle= True)
    # thumb_input = thumb_input.tolist()
    thumb_path= thumb_nail_main(thumb_input)
    print("thumbnail successed!!")
    

    return {
        'video_image': thumb_path, 
        'video_path': save_path,
        'video_tag': hashtag,
        'user_ID': user_ID
    }



if __name__ == '__main__': 
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)
    # app.run(debug=True)
    predict()


