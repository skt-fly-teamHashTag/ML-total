import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import torch
import matplotlib.pyplot as plt  
import time 

from helpers import init_helper, adot_vsumm_helper, bbox_helper, video_helper
from modules.model_zoo import get_model

import moviepy.editor as mp
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

def get_audio_text(path, i, r):
    '''
    구간 내의 영상의 음성을 텍스트로 추출후 리턴 
    '''
    # file_path = f"../custom_data/audio_text/stt_{i}.txt"
    # f = open(file_path, 'w')
    out_text = [] 
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path) 
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk 
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened, language = 'ko-KR')
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                # f.write(text+'\n')
                out_text.append(text)
    # return the text for all chunks detected
    return out_text


def audio_cps_split(fps, audio_path, ws_cps, ws_audio_dir):
    '''
    프레임넘버에 따라 음성의 시간을 파악한 뒤,
    해당 시간구간만큼 음성을 추출
    ''' 
    sound = mp.AudioFileClip(audio_path) 
    i = 0
    start_time = 0 
    end_time = 0  
    for frame_num in ws_cps: 
        ws_audio_path = f"ws_audio_{i}.wav"
        end_time = round(frame_num/fps, 2)
        print(f"ws_cps time: {end_time}s")

        sub_sound = sound.subclip(start_time, end_time)
        sub_sound.write_audiofile(os.path.join(ws_audio_dir, ws_audio_path))
        start_time = end_time 

        i +=1
    ws_audio_path = f"ws_audio_{i}.wav"
    end_time = sound.duration
    print(f"ws_cps time: {end_time}s")
    sub_sound = sound.subclip(start_time, end_time)
    sub_sound.write_audiofile(os.path.join(ws_audio_dir, ws_audio_path))

    # 각 음성구간 마다 텍스트로 추출
    r = sr.Recognizer()
    len_sound = len(ws_cps)+1
    total_text = [] 
    for i in range(len_sound): 
        ws_audio_path = f"ws_audio_{i}.wav"
        path = os.path.join(ws_audio_dir, ws_audio_path)
        total_text.append(get_audio_text(path, i, r))

    return total_text 


def video_shot_main(source):
    '''
    input:
       - source: lst= ['src1', 'src2', 'src3', ..]
    '''
    args = init_helper.get_arguments()

    # init setting #
    ckpt_path = '../models/pretrain_ab_basic/checkpoint/summe.yml.0.pt'
    # source = '../custom_data/videos/shasha_drawing0.mp4'
    sample_rate = 15 
    
    audio_path = '../custom_data/audio/audio.wav' #추출 오디오 저장 위치 
    ws_audio_dir = '../custom_data/audio' #오디오 구간 분리 seg audio 저장 위치 

    if not os.path.exists('../custom_data/audio'):
        os.makedirs('../custom_data/audio')

    # load model
    print('Loading DSNet model ...')
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    state_dict = torch.load(ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # Video Preprocessor 
    # 입력받은 여러 영상들을  
    #  (1) 하나의 영상으로 병합 & 오디오 추출->audio_path 경로에 (.wav) 저장   
    #  (2) 영상 sampling & feature extraction 
    #  (3) 영상 cps -> algorithm: cpd_auto, 음성 ws_cps 구간 나누기 ->algorithm: cpd_nonlin  
    #  (4) ObjectDetection (yolov5)
    # -----------------------------
    '''
    ## feature extracted of frame 
    n_frames: # of total video frames 
    seq: extracted frame feature sequence 
    cps: change point segment 
    nfps: # of frames per segment 
    picks: position of frames at original video sequence 

    ## audio change point detected
    ws_cps: audio split point frame number [int, int, ..]
    fps: frame per second 

    ## object detection 
    frame_obj_lst: [(fnum, {'2':0, '1':1}), (fnum, {'3':2, '4': 3}), ..] 
     -> sample rate마다 추출된 frame이 5번 추출되면 한번 샘플링 (=약 1.25s마다 한번 샘플링)
    '''
    print('Preprocessing source video ...')
    video_proc = video_helper.VideoPreprocessor(sample_rate, audio_path)
    n_frames, seq, cps, nfps, picks, ws_cps, fps, frame_obj_lst = video_proc.run(source) #seq:extracted features from CNN, change points: 세그먼트 구분
    seq_len = len(seq)

    # 음성 cps 구간 분리 
    total_stt = audio_cps_split(fps, audio_path, ws_cps, ws_audio_dir)

    ## 구간별 object detection으로 검출된 물체 counting dict 구성 
    ws_obj_lst= [] #구간별, 각 구간 내의 프레임들의 객체 딕셔너리 
    tmp_dict = {} 
    k = 0
    cur_fnum = ws_cps[k]
    for fnum, obj_dict in frame_obj_lst:
        if fnum >= cur_fnum:
            ws_obj_lst.append(tmp_dict)
            k += 1
            if k >= len(ws_cps):
                cur_fnum = n_frames
            else:
                cur_fnum = ws_cps[k]
            tmp_dict = {}
        print(f"fnum: {fnum}, cur_frame: {cur_fnum}, k: {k}")
        for key in obj_dict.keys():
            if key in tmp_dict: 
                tmp_dict[key] += obj_dict[key]
            else:
                tmp_dict[key] = obj_dict[key]

    ws_obj_lst.append(tmp_dict)

    return total_stt, ws_obj_lst, seq, model, cps, n_frames, nfps, picks, ws_cps

def bgm(vlog_path, bgm_video_path, ws_cps, cat, fps, n_frames):
    
    videoclip = mp.VideoFileClip(vlog_path) #출력 브이로그 영상 
    audioclip = mp.AudioFileClip(f"../custom_data/music/music0.mp3") 

    '''
    if len(cat) == 1:
        audioclip = mp.AudioFileClip(f"../custom_data/music/music{cat[0]}.mp3") #합성할 배경음 위치
        new_audioclip = mp.afx.audio_loop(audioclip, duration=videoclip.duration)
        # audio.write_audiofile("audio.mp3") ##생성된 음성 파일 저장하기 
    else: #두 카테고리 중에 높은 카테고리의 음악을 배경으로 튼다 
        audioclip0 = mp.AudioFileClip(f"../custom_data/music/music{cat[0]}.mp3")
        audioclip1 = mp.AudioFileClip(f"../custom_data/music/music{cat[1]}.mp3")
        sub_audios = []
        for i in range(len(ws_class)):
            if i == 0:
                s_frame = 0 
            else:
                s_frame = ws_cps[i-1]
            if i == len(ws_class)-1:
                e_frame = n_frames
            else:
                e_frame = ws_cps[i]
            s_time = round(s_frame/fps, 2)
            e_time = round(e_frame/fps, 2)
            dur = e_time - s_time 
            
            s_frame = e_frame

            if ws_class == 0: 
                sub_audios.append(audioclip0.subclip(0, dur))
            else: 
                sub_audios.append(audioclip1.subclip(0, dur))
        
        new_audioclip = mp.CompositeAudioClip(sub_audios)
        '''
    new_audioclip = mp.afx.audio_loop(audioclip, duration=videoclip.duration)
    videoclip.audio = new_audioclip
    videoclip.write_videofile(bgm_video_path)

    return None

def makeSumm(seq, model, cps, n_frames, nfps, picks, source, save_path, ws_score, ws_cps, total_stt, cat):
    device = "cpu" #else, "cuda"
    seq_len = len(seq)
    nms_thresh = 0.5

    print('Predicting summary ...')
    with torch.no_grad():
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

        pred_cls, pred_bboxes = model.predict(seq_torch) #features의 score 평가 

        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
        
        """Convert predicted bounding boxes to summary"""
        pred_summ, thumb_nail, thumb_nail_scores = adot_vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks, ws_score, ws_cps) ##knapsac 알고리즘으로 키샷 추출 

    print('Writing summary video ...')

    # load original video 
    cap = cv2.VideoCapture(source[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 영상 저장 경로 존재 여부 확인 
    save_dir = "/".join(save_path.split("/")[:-1])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # 배경음 없는 중간 영상 임시 저장 
    tmp_dir = "../output/tmp"
    tmp_name = "non_bgm_vlog.mp4"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_path = os.path.join(tmp_dir, tmp_name)
    

    # create summary video writer 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    thumb_frames = []
    frame_idx = 0
    # summ_frame_idx = 0
    # text_duration = 0 # 자막 샘플링마다 3초간 유지하도록 변수 초기화 
    start = time.time()

    # k = 0 # 현재 프레임이 어떤 ws_cps 구간에 포함되어있는지 인덱스 
    # prev_fnum = 0 
    # cur_fnum = ws_cps[k]
    # text = ""
    # prev_text = ""
    # fontpath = "fonts/gulim.ttc" #자막 폰트 설정 
    # font = ImageFont.truetype(fontpath, 50)
    for i in range(len(source)):
        video_path = source[i]
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        assert cap is not None, f'Cannot open video: {video_path}'

        while True:
            ret = cap.grab()
            ret, frame = cap.retrieve()
            
            # h, w, _ = frame.shape

            if not ret:
                break

            # # 현재 프레임이 ws_cps 몇번째 구간인지 
            # if frame_idx > cur_fnum: 
            #     k+=1
            #     if k >= len(ws_cps):
            #         prev_fnum = cur_fnum 
            #         cur_fnum = n_frames
            #     else:
            #         prev_fnum = cur_fnum 
            #         cur_fnum = ws_cps[k]

            # 요약 & 자막 
            if pred_summ[frame_idx]:
                # # 자막달기 (7초에 한번씩 자막 샘플링, 3초 유지)
                # # 5초마다 자막 샘플링 
                # if summ_frame_idx % (7*fps) == 0:
                #     portion = round((frame_idx-prev_fnum)/(cur_fnum - prev_fnum), 2)
                #     idx = int(len(total_stt[k])*portion)
                #     text = total_stt[k][idx]
                #     while text==prev_text: 
                #         idx += 1
                #         if idx >= len(total_stt[k]):
                #             text =""
                #             break 
                #         prev_text = text
                #         text = total_stt[k][idx]
                #     prev_text = text
                #     text_duration = 0 

                # # 자막 샘플링 한 뒤 3초간만 유지 
                # if text_duration <= (3*fps):
                #     # size, _ =cv2.getTextSize(text,font,1,2) #텍스트 사각형 : (폰트 스케일, 두께)에 따라
                #     # cv2.putText(frame, text, (int(w/2), h-4*size[1]), font, 3, (255, 255, 255))
                #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     pil_im = Image.fromarray(frame_rgb)
                #     draw = ImageDraw.Draw(pil_im)
                #     # 자막 텍스트 넣기 
                #     textw, texth = draw.textsize(text, font= font)
                #     draw.text((int(w/2-textw/2), int(h*0.95-texth)), text, font= font)
                #     frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

                out.write(frame)
                # summ_frame_idx += 1
                # text_duration += 1
                   

        
            if thumb_nail[frame_idx]:
                (cps_score, frame_score) = thumb_nail_scores[frame_idx]
                thumb_frames.append([frame, cps_score, frame_score])

            frame_idx += 1
    end = time.time()
    print(f"writing time: {round(end-start)}s")

    out.release()
    cap.release()

    bgm(tmp_path, save_path, ws_cps, cat, fps, n_frames)

    return thumb_frames


if __name__ == '__main__':
    ## video summary & save 
    print('*** start video summary ***') 
    thumb_input, caption_images = video_shot_main() #[[image, cps_score, frame_score], ...] 
    # print(f'len(thumbnail_images): {len(thumb_input)}, len(caption_images): {len(caption_images)}')



