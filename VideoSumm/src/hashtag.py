import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# 명사 문장 추출
class SentenceTokenizer(object):
    def __init__(self):
        self.okt = Okt()
        self.stopwords = ['남자', '여자']
        
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.okt.nouns(sentence) if noun not in self.stopwords and len(noun) > 1]))
        
        return nouns

# TF-IDF 모델 생성과 그래프 생성
class GraphMatrix(object):
    def __init__(self):
        self.cnt_vec = CountVectorizer()
    
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}

# TextRank 알고리즘
class Rank(object):
    def get_ranks(self, graph, d=0.85):
        A = graph
        matrix_size = A.shape[0]
        for idx in range(matrix_size):
            A[idx, idx] = 0
            link_sum = np.sum(A[:, idx])
            if link_sum != 0:
                A[:, idx] /= link_sum
            A[:, idx] *= -d
            A[idx, idx] = 1
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)
        
        return {idx: r[0] for idx, r in enumerate(ranks)}

# TextRank Class
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        self.nouns = self.sent_tokenize.get_nouns(text)
        
        self.graph_matrix = GraphMatrix()
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        
        self.rank = Rank()
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
    
    def keywords(self, word_num=3):
        keywords = []
        index = []
        
        for idx in self.sorted_word_rank_idx[:word_num]:
            index.append(idx)
        
        for idx in index:
            keywords.append(self.idx2word[idx])
        
        keywords[0] = keywords[0]+'브이로그'
        
        return keywords