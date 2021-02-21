# 在feat_preproc之后执行，从feat_preproc中获取初步生成的score_record，选择出关键帧以及关键片段，重新保存

import os
import numpy as np
import cv2
import math
import json
from tools.knapsack_iter import knapSack

SCORE_PATH = r'/data/linkang/VHL_GNN/tvsum_score_record.json'
SEGINFO_PATH = r'/data/linkang/VHL_GNN/tvsum_segment_info.json'
KEY_FRAME_RATIO = 0.35 # 整体上大约有33%的帧标记为keyframe

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

def frame2shot(vid,segment_info,scores):
    # 输入N*vlength的帧得分，以及对应视频的分段情况，输出同样形状的keyshot_labels
    # keyshot_labels将所有被选入summary的帧标记为1，其他标记为0
    cps = np.array(segment_info[vid])
    keyshot_labels = []
    for i in range(len(scores)):
        y = scores[i]
        lists = [(y[cps[idx]:cps[idx + 1]], cps[idx]) for idx in range(len(cps) - 1)]
        segments = [tuple([np.average(i[0]), len(i[0]), i[1]]) for i in lists]
        value, weight, start = zip(*segments)
        max_weight = int(0.15 * len(y))
        chosen = knapSack(max_weight, weight, value, len(weight))
        keyshots = np.zeros(len(y))
        chosen = [int(j) for i, j in enumerate(chosen)]
        for i, j in enumerate(chosen):
            if (j == 1):
                keyshots[start[int(i)]:start[int(i)] + weight[int(i)]] = 1
        keyshot_labels.append(keyshots)
    keyshot_labels = np.array(keyshot_labels).squeeze()
    return keyshot_labels

def label_process(score_record, segment_info, keyframe_ratio):
    # 1. 从scores_avg中根据一定的比例选择出keyframe
    # 2. 根据分段信息与标注，生成每个标注对应的summary，以帧的二分类标签形式呈现

    # select keyframe
    vids = list(score_record.keys())
    key_ratios = []
    for vid in vids:
        scores_avg = np.array(score_record[vid]['scores_avg'])
        score_list = list(scores_avg)
        score_list.sort(reverse=True)
        threshold = score_list[math.ceil(len(scores_avg) * keyframe_ratio)]  # 得分大于门限的帧设为keyframe
        labels = (scores_avg > threshold).astype(int)
        score_record[vid]['labels'] = labels
        key_ratios.append(np.mean(labels))
    key_ratios = np.array(key_ratios)
    # print(keyframe_ratio,np.min(key_ratios),np.max(key_ratios),np.mean(key_ratios))

    # convert frame score to keyshot
    for vid in vids:
        scores = np.array(score_record[vid]['scores'])  # 20*vlength
        keyshot_labels = frame2shot(vid,segment_info,scores)  # 20*vlength
        score_record[vid]['keyshot_labels'] = keyshot_labels
    return score_record

if __name__ == '__main__':
    with open(SCORE_PATH,'r') as file:
        score_record = json.load(file)
    with open(SEGINFO_PATH, 'r') as file:
        segment_info = json.load(file)

    score_record_new = label_process(score_record, segment_info, KEY_FRAME_RATIO)

    with open(SCORE_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)

    print('Done !')