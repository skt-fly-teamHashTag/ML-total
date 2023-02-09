import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt  

from helpers import init_helper, adot_vsumm_helper, bbox_helper, video_helper
from modules.model_zoo import get_model

def video_shot_main(source):
    args = init_helper.get_arguments()

    # init setting #
    ckpt_path = '../models/pretrain_ab_basic/checkpoint/summe.yml.0.pt'
    # source = '../custom_data/videos/shasha_drawing0.mp4'
    sample_rate = 15 
    save_path = '../output/vlog.mp4'
    nms_thresh = 0.5

    # load model
    print('Loading DSNet model ...')
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    state_dict = torch.load(ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # load video
    print('Preprocessing source video ...')
    video_proc = video_helper.VideoPreprocessor(sample_rate)
    n_frames, seq, cps, nfps, picks = video_proc.run(source) #seq:extracted features from CNN, change points: 세그먼트 구분
    seq_len = len(seq)

    print('Predicting summary ...')
    with torch.no_grad():
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)

        pred_cls, pred_bboxes = model.predict(seq_torch) #features의 score 평가 

        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
        
        """Convert predicted bounding boxes to summary"""
        pred_summ, thumb_nail, thumb_nail_scores = adot_vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks) ##knapsac 알고리즘으로 키샷 추출 

    print('Writing summary video ...')

    # load original video 
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create summary video writer 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    thumb_frames = []
    caption_frames = [] 

    frame_idx = 0
    n_cps_framse = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if pred_summ[frame_idx]:
            out.write(frame)
            n_cps_framse+= 1
            if n_cps_framse %60 == 1:
                caption_frames.append([frame, frame_idx])

        else: 
            n_cps_framse = 0 

        if thumb_nail[frame_idx]:
           (cps_score, frame_score) = thumb_nail_scores[frame_idx]
           thumb_frames.append([frame, cps_score, frame_score])

        frame_idx += 1

    out.release()
    cap.release()

    return thumb_frames, caption_frames



if __name__ == '__main__':
    ## video summary & save 
    print('*** start video summary ***') 
    thumb_input, caption_images = video_shot_main() # [[image, cps_score, frame_score], ...] 
    # print(f'len(thumbnail_images): {len(thumb_input)}, len(caption_images): {len(caption_images)}')



