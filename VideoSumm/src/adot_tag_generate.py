import random
import numpy as np

from transformers import BertModel
from keybert import KeyBERT

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

import gensim
import pickle

class GetKeyword():
  def __init__(self):
    # KeyBERT 모델을 통한 keyword 추출
    model = BertModel.from_pretrained('skt/kobert-base-v1')
    self.kw_extractor = KeyBERT(model)

  def get_keyword(self,total_stt, max_idx) -> str:
    # 스코어가 높은 구간의 문장들을 합침
    sentence = ' '.join(total_stt[max_idx])
    
    keywords = self.kw_extractor.extract_keywords(
        sentence, 
        keyphrase_ngram_range=(1, 1), 
        stop_words=None, 
        top_n=5) # 5개만 추출하도록 함
    
    # 추출된 키워드 문장으로 합침
    text = ''
    for keyword, score in keywords:
        text += ' ' + keyword
    
    # 명사 키워드 추출(고민할 부분: 명사를 추출할 필요가 있는가?)
    n_temp = []

    kiwi = Kiwi()
    stopwords = Stopwords()
    
    for word in kiwi.tokenize(text, stopwords=stopwords):
        if word[1][:2] == 'NN' and len(word[0]) > 1:
            n_temp.append(word[0])
    
    try:
        temp = n_temp[0] # 첫 번째 명사 키워드
    except:
        temp = keywords[0][0] # 첫 번째 키워드
    
    return temp
  
  def __call__(self,sentences: list,max_idx: int) -> str:
    """
    input : 
        - total_stt : [[sentences1], [sentences2], ...]
        - max_idx : int
    output : 
        - temp : str
    """
    result = self.get_keyword(sentences,max_idx)
    return result

class GetHashtag():
  def __init__(self,path='pickle_data'):
      self.ko_model = gensim.models.Word2Vec.load('ko.bin')
      # hash table
      with open(f'{path}/hash_table.pickle', 'rb') as f:
          self.hash_table = pickle.load(f)
      
      # common tag
      with open(f'{path}/common_tag.pickle', 'rb') as f:
          self.common_tag = pickle.load(f)
      
      # image dic
      with open(f'{path}/dic.pickle', 'rb') as f:
          self.dic = pickle.load(f)
      
      # image table
      with open(f'{path}/img_table.pickle', 'rb') as f:
          self.img_table = pickle.load(f)
      self.label = ['가족', '스터디', '뷰티', '반려동물', '운동/스포츠', 
              '음식', '여행', '연애/결혼', '문화생활', '직장인']

  def get_hashtag(self,user_cat: list, ws_obj_lst: list, max_idx: int, temp: str) -> list:
      # hash_tag list
      hashtag = []
      
      # hash table 사용
      if len(user_cat) == 1:
          hashtag.extend(random.sample(self.hash_table[self.label[user_cat[0]]],2))
      elif len(user_cat) == 2:
          hashtag.append(random.choice(self.hash_table[self.label[user_cat[0]]]))
          hashtag.append(random.choice(self.hash_table[self.label[user_cat[1]]]))
      
      # common tag 사용
      hashtag.append(random.choice(self.common_tag))
      
      # image dic 유사도 비교
      new_dic = {}
      
      obj = ws_obj_lst[max_idx]
      
      for x, cnt in obj.items():
          try:
              if self.ko_model.wv.similarity(temp, self.dic[x]) >= 0.3: # 유사도
                  new_dic[x] = cnt
          except:
              pass
      
      dic_list = sorted(new_dic, key=lambda x: new_dic[x], reverse=True) # 정렬
      
      # image table 사용
      for x in dic_list[:2]: # 상위 스코어 이미지 2개에 대해서 해시태그 추출
          hashtag.append(random.choice(self.img_table[x]))
      
      random.shuffle(hashtag)
      
      return hashtag
  def __call__(self,user_cat: list, ws_obj_lst: list, max_idx: int, temp: str):
      """
      input : 
          - user_cat : [cat1_int, cat2_int]
          - ws_obj_lst : [{'1': 2, '3': 5}, {'1': 3, '4': 8}, ...]
          - max_idx : int
          - temp : str
          - path : str
      output : 
          - hashtag : ['hash1', 'hash2', ...]
      """
      result = self.get_hashtag(user_cat,ws_obj_lst,max_idx,temp)
      return result

def get_section_score(user_cat, total_stt,predict):
    """
    input : 
        - user_cat : [cat1_int, cat2_int]
        - total_stt : [[sentences1], [sentences2], ...]
    output : 
        - ws_score : [score1, score2, ...]
    """
    ws_score = [] #각 구간별 두 카테고리의 output의 합
    ws_class = []

    if len(user_cat) == 1: # 카테고리가 1개인 경우
        for sent_list in total_stt: # if ws_score is none, fill the -1 
            if len(sent_list) == 0:
                ws_score.append(-1)
            else: # else case is by predict function
                output = predict(sent_list)
                idx = user_cat[0]
                ws_score.append(output[idx])
    elif len(user_cat) == 2: # 카테고리가 2개인 경우
        for sent_list in total_stt:
            if len(sent_list) == 0: # this is conner case so we fill the -1
                ws_score.append(-1)
                ws_class.append(-1)
            else:
                output = predict(sent_list)
                idx1, idx2 = user_cat[0], user_cat[1]
                ws_score.append(output[idx1] + output[idx2])
                if output[idx1]> output[idx2]:
                    ws_class.append(0)
                else: 
                    ws_class.append(1)

    return ws_score, ws_class

