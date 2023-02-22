from os import PathLike
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms, models
import time
from kts.cpd_auto import cpd_auto, cpd_nonlin
import moviepy.editor as mp
from adot_detection import load_detection_model, detection_run

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
        self.model = self.model.cpu().eval()

    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            feat = self.model(batch.cpu())
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat /= linalg.norm(feat) + 1e-10
        return feat

class VideoPreprocessor(object):
    def __init__(self, sample_rate: int, output_audio_path: str) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate
        self.output_audio_path = output_audio_path

    def get_features(self, video_path_lst: PathLike):
        '''
        gey video frames features & extracted audioclip 
        ''' 
        # yolo detectin 
        weights =  "yolov5s.pt"
        source =[]
        frame_obj_lst = []
        model, stride, names, pt = load_detection_model(weights = weights) # load model 
        # frame feature 
        features = []
        n_frames = 0
        audio_clips = [] 
        tmp_num = 0 
        for i in range(len(video_path_lst)):
            video_path = video_path_lst[i]
            # video_path = Path(video_path)

            videoclip = mp.VideoFileClip(video_path) 
            audioclip = videoclip.audio #영상의 오디오 추출 
            audio_clips.append(audioclip)
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)

            assert cap is not None, f'Cannot open video: {video_path}'
            
            start = time.time()
            while True:
                # ret, frame = cap.read() #read() = grab() + retrieve() 
                ret = cap.grab()
                ret, frame = cap.retrieve()

                if not ret:
                    break
                
                if n_frames % self.sample_rate == 0:
                    # object detection 
                    tmp_num += 1
                    if tmp_num % 5 == 0: 
                        obj_dict = detection_run(source = "", model= model, stride = stride, names = names, pt = pt, im0 =frame)
                        frame_obj_lst.append((int(n_frames/self.sample_rate), obj_dict))
                        tmp_num = 0

                    # CNN feature extraction 
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                    feat = self.model.run(frame)
                    features.append(feat)
          
                n_frames += 1

            end = time.time()
            print(f"cap time: {(end-start)//60}min {(end-start)%60}s")
            cap.release()

        features = np.array(features)

        final_audioclip = mp.concatenate_audioclips(audio_clips) #영상들의 오디오 하나로 병합 
        final_audioclip.write_audiofile(self.output_audio_path) #음성파일 저장 
        print("#"*5,"audio extracted finished!!" ,"#"*5)

        return n_frames, features, fps, frame_obj_lst

    def kts(self, n_frames, features, len_videos):
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        # compute change points using KTS
        max_ncp = seq_len -1
        kernel = np.matmul(features, features.T)
        ws_change_points, _ = cpd_nonlin(kernel, len_videos + 3) #사용자에게 입력받는 각 하위 영상마다 구간별로 주제 끊기 
        ws_change_points *= self.sample_rate
        change_points, _ = cpd_auto(kernel, max_ncp, 1, verbose=False)
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg, picks, ws_change_points

    def run(self, video_path: PathLike):
        n_frames, features, fps, obj = self.get_features(video_path) 
        print(f"--- done get feature --- n_frames:{n_frames}, len features: {len(features)}") 
        cps, nfps, picks, ws_cps = self.kts(n_frames, features, len(video_path)) 
        print(f"# of change points: {len(cps)}") 
        print("--- done kts ---") 
        return n_frames, features, cps, nfps, picks, ws_cps, fps, obj


