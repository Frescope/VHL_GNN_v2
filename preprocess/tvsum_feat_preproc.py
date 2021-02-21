# 前处理步骤：首先截取帧并读取对应位置的帧标签，记录所有标签以及标签的平均值，不做归一化
# 读取视频，按照固定帧率截取图像，保存为jpg格式，对图像大小不作要求，统一留到特征提取时处理
# 同时汇总每个类别的所有视频id
# 使用myenv1环境

import os
import numpy as np
import cv2
import math
import json

VIDEO_BASE = r'/data/linkang/tvsum50/video/'
FRAME_BASE = r'/data/linkang/tvsum50/frame_2fps/'
ANNO_PATH = r'/data/linkang/tvsum50/anno_info.json'
INFO_PATH = r'/data/linkang/tvsum50/video_info.json'
SCORE_PATH = r'/data/linkang/VHL_GNN/tvsum_score_record.json'
VCAT_PATH = r'/data/linkang/VHL_GNN/tvsum_video_category.json'
FRAME_RATE = 2  # 2fps

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def load_info(info_path, anno_path):
    # 加载标签以及视频元数据
    with open(info_path,'r') as file:
        video_info = json.load(file)
    with open(anno_path,'r') as file:
        anno_info = json.load(file)
    return video_info, anno_info

def frame_capture(anno_info,vid,video_base,frame_base):
    video_path = video_base + vid + '.mp4'
    frame_dir = frame_base + vid + r'/'
    if os.path.isdir(frame_dir):
        print('Directory already exists !')
        return
    else:
        os.makedirs(frame_dir)

    # get annotations
    annos = []  # 关于这一id的所有标注
    for id in list(anno_info.keys()):
        anno_record = anno_info[id]
        if anno_record['vid'] == vid:
            annos.append(anno_record['anno'])
    annos = np.array(annos)

    # capturing
    scores = np.zeros((len(annos),0))
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_interval = round(fps / FRAME_RATE)  # 截取帧的间隔
    rval, frame = vc.read()
    frame_count = 0
    while rval:
        if frame_count % frame_interval == 0:
            frame_time = round(frame_count / fps * 10)  # 帧对应的秒数*10
            frame_path = frame_dir + str(frame_time).zfill(6) + '.jpg'
            cv2.imwrite(frame_path, frame)
            score_frame = annos[:,frame_count:frame_count + 1]  # 这一帧对应的所有标注
            scores = np.hstack((scores,score_frame))
        if frame_count % 1000 == 0 and frame_count > 0:
            print('Frames: ',frame_count)
        rval, frame = vc.read()
        frame_count += 1
    vc.release()
    print('Frames & Scores Extracted: ',vid,frame_count,scores.shape)
    # feat = np.load(r'/data/linkang/VHL_GNN/tvsum_feature_googlenet_2fps/'+vid+r'_googlenet_2fps.npy')
    # print(feat.shape)
    return scores

if __name__ == '__main__':
    count = 0
    video_info, anno_info = load_info(INFO_PATH, ANNO_PATH)
    score_record = {}
    video_category = {}
    for root, dirs, files in os.walk(VIDEO_BASE):
        for file in files:
            vid = file.split('.mp4')[0]
            print('-'*20,count,vid,'-'*20)
            score_record[vid] = {}
            scores = frame_capture(anno_info, vid,VIDEO_BASE,FRAME_BASE)
            # recording
            score_record[vid]['scores'] = scores
            score_record[vid]['scores_avg'] = np.mean(scores, axis=0)
            score_record[vid]['frame_count'] = video_info[vid]['frame_count']
            score_record[vid]['fps'] = video_info[vid]['fps']
            category = video_info[vid]['category']
            score_record[vid]['category'] = category
            if category not in video_category.keys():
                video_category[category] = []
            video_category[category].append(vid)  # 登记每个类别的所有id
            count += 1

    with open(SCORE_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)
    with open(VCAT_PATH, 'w') as file:
        json.dump(video_category, file, cls=NpEncoder)
    print('Done !')