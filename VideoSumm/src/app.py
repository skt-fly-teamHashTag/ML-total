import cv2
import numpy as np
import torch
import torchvision 
import matplotlib.pyplot as plt  
from modules.model_zoo import get_model
from dsnet_main import video_shot_main
from hashtag import TextRank
from qwer import qwe
import urllib.request

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
    

def save_video(video_url) :
    saveName = '../origin_video/video.mp4'

    urllib.request.urlretrieve(video_url,saveName)
    print("저장완료")

    return saveName


@app.route('/video_summary', methods=['POST'])
def predict():
    
    data = request.get_json() 
    user_ID = data['user_id'] 
    video_src = data['video_origin_src'] 

    video_path = save_video(video_src)

    ##영상요약 
    # video_src = '../custom_data/videos/test11_shopping.mp4'
    thumb_input, caption_images = video_shot_main(video_path) #thumb_input: type==list 
    print(f'len(thumbnail_images): {len(thumb_input)}, len(caption_images): {len(caption_images)}')
    print("video summary successed!!")
    ##썸네일 
    # thumb_input = np.load('../output/test/test7_class_thumb_9.npy', allow_pickle= True)
    # thumb_input = thumb_input.tolist()
    thumb_path= thumb_nail_main(thumb_input)
    print("thumbnail successed!!")
    ##캡셔닝 
    # caption_images = np.load('/content/drive/MyDrive/skt-flyai /skt-fly-teamHashTag/DSNet/output/test/test10_drawing_thumb_23.npy', allow_pickle = True)
    sentences = caption_expansion(caption_images)
    print("captioning successed !!")
    # print(sentences)
    ##해시태그 추출 
    hashtag_output = hashtag_main(sentences)
    print("해시태그: ", hashtag_output)
    print("hashtag extracted finished!!")

    return {
        'video_image': thumb_path, 
        'video_path': '../ouput/vlog.mp4',
        'video_tag': hashtag_output,
        'user_ID': user_ID
    }



if __name__ == '__main__': 
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000)
    # app.run(debug=True)
    predict()


