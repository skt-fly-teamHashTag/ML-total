from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms, models

from kts.cpd_auto import cpd_auto ##kernel temporal segmentation with autocalibration


class FeatureExtractor(object):
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = models.googlenet(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.cuda().eval()

    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            feat = self.model(batch.cuda())
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat /= linalg.norm(feat) + 1e-10
        return feat


class VideoPreprocessor(object):
    def __init__(self, sample_rate: int) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate
        '''
        VideoPreprocessor():
            get_feature: frame sampling & googlenet 특징 추출 
            kts(kernel temporal segment): change point 에 따른 프레임 구간 스플릿 

        '''
    def get_features(self, video_path: PathLike):
        '''
        1) video path 에서 영상을 load 한 뒤, sampling rate 마다 frame sampling
        2) googlenet을 특징 추출기로 사용해 프레임들의 특징 추출 

        return
            - n_frames: # of smapling frame 
            - features: 샘플링 된 frame들의 feature 
        '''
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        features = []
        n_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0: #smapling rate 마다 frame 추출 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = self.model.run(frame) ##googlenet을 거쳐 추출된 feature ###프레임의 특징 추출기 
                features.append(feat)

            n_frames += 1

        cap.release()

        features = np.array(features)
        return n_frames, features

    def kts(self, n_frames, features):
        seq_len = len(features) #총 샘플링된 프레임의 갯수 
        picks = np.arange(0, seq_len) * self.sample_rate # 추출된 sample frame의 origin 프레임 상의 위치 
        
        # compute change points using KTS
        kernel = np.matmul(features, features.T) #내적을 통해 kernel 생성 
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False) #Square kernel, 최대cp갯수, 
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg, picks

    def run(self, video_path: PathLike):
        '''
        cps: Change points, 2D matrix, each row contains a segment.
        nfps: Number of frames per segment.
        picks: Positions of subsampled frames in the original video.
        '''
        n_frames, features = self.get_features(video_path) 
        cps, nfps, picks = self.kts(n_frames, features) 
        return n_frames, features, cps, nfps, picks 



