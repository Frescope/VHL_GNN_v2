# 生成帧得分，读取每个视频的所有标注，将相同位置的得分取平均作为帧得分，记录帧得分即可
# 关键帧的标注留在模型训练阶段进行，根据训练情况调整关键帧比例
# 为了避免随机性，在这里设定一个初始的划分索引加入标签文件中
# 这里的帧得分是所有帧的得分，降采样之后的帧对应的得分在获取帧时进一步确定

import numpy as np
import os
import json
import math
from tools.old_Inception_feat_extract import hello

ANNO_PATH = r'/data/linkang/tvsum50/anno_info.json'
INFO_PATH = r'/data/linkang/tvsum50/video_info.json'

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

def label_build(vid,anno_info):
    annos = []  # 关于这一id的所有标注
    for id in list(anno_info.keys()):
        anno_record = anno_info[id]
        if anno_record['vid'] == vid:
            annos.append(anno_record['anno'])
    annos = np.array(annos)
    annos_mean = np.mean(annos, axis=0)
    score = (annos_mean - np.min(annos_mean)) / (np.max(annos_mean) - np.min(annos_mean)) + 1e-6
    return score

if __name__ == '__main__':
    with open(INFO_PATH,'r') as file:
        video_info = json.load(file)
    vids = list(video_info.keys())
    with open(ANNO_PATH,'r') as file:
        anno_info = json.load(file)

    score_record = {}
    video_category = {}
    for vid in vids:
        score_record[vid] = {}
        score = label_build(vid,anno_info)
        score_record[vid]['score'] = score
        score_record[vid]['frame_count'] = video_info[vid]['frame_count']
        score_record[vid]['fps'] = video_info[vid]['fps']
        category = video_info[vid]['category']
        score_record[vid]['category'] = category
        if category not in video_category.keys():
            video_category[category] = []
        video_category[category].append(vid)  # 登记每个类别的所有id

    with open(r'/data/linkang/VHL_GNN/tvsum_score_record.json', 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)
    with open(r'/data/linkang/VHL_GNN/tvsum_video_category.json', 'w') as file:
        json.dump(video_category, file, cls=NpEncoder)
    print('Done !')