import numpy as np
from konlpy.tag import Okt
from transformers import BertModel
from keybert import KeyBERT
# import nltk
# from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# 명사 문장 추출(한국어)
class SentenceTokenizer(object):
    def __init__(self):
        self.okt = Okt()
        self.stopwords = ['남자', '여자', '사람', '가래', '여성']
        
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.okt.nouns(str(sentence)) if noun not in self.stopwords and len(noun) > 1]))
        
        return nouns

# # 명사 문장 추출(영어)
# class SentenceTokenizer(object):
#     def __init__(self):
#         self.lmtzr = WordNetLemmatizer()

#     def get_nouns(self, sentences):
#         nouns = []
#         for sentence in sentences:
#             tokenized = nltk.word_tokenize(sentence.lower())
#             tokens = [word for (word, pos) in nltk.pos_tag(tokenized) if (pos[:2] == 'NN')]
#             nouns.append(' '.join([self.lmtzr.lemmatize(token) for token in tokens]))
        
#         return nouns

# KeyBERT 알고리즘
class Rank(object):
    def get_keywords(self, sentences):
        sentence = ' '.join(sentences)
        
        model = BertModel.from_pretrained('skt/kobert-base-v1')
        # model = 'distiluse-base-multilingual-cased-v1'
        kw_extractor = KeyBERT(model)

        keywords = kw_extractor.extract_keywords(
            sentence, 
            keyphrase_ngram_range=(1, 1), 
            stop_words=None, 
            top_n=50)
        
        return keywords

# KeyBERT hashtag Class
class KeyBERTModel(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        self.nouns = self.sent_tokenize.get_nouns(text)
        
        self.rank = Rank()
        self.keywords = self.rank.get_keywords(self.nouns)
    
    def get_hashtags(self, word_num=3):
        hashtags = []
        
        for keyword, score in self.keywords[:word_num]:
            hashtags.append(keyword)
        
        hashtags[0] = hashtags[0]+'브이로그'
        
        return hashtags