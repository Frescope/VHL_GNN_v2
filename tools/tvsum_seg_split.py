# 使用KTS方法，基于googlenet特征做cpd分段，即对于gt和预测，分段都是相同的
# 根据视频长度限定分段个数，分段数量上限为视频时长除以2s
# 统一按照每两帧计为一秒，可能会导致部分视频被拉长
# 使用torch环境

import numpy as np
import os
import json
import math
from tools.cpd_auto import cpd_auto, estimate_vmax

np.random.seed(1997)  # 限定seed保证可复现性

FEATURE_BASE = r'/data/linkang/VHL_GNN/tvsum_feature_googlenet_2fps/'
SCORE_PATH = r'/data/linkang/VHL_GNN/tvsum_score_record.json'
SEGINFO_PATH = r'/data/linkang/VHL_GNN/tvsum_segment_info.json'
FRAME_RATE = 2

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

def load_score(score_path):
    with open(score_path, 'r') as file:
        score_record = json.load(file)
    return score_record

def load_feature(score_record, feature_base):
    vids = list(score_record.keys())
    features = {}
    for vid in vids:
        feature_path = feature_base + vid + r'_googlenet_2fps.npy'
        feature = np.load(feature_path)
        print(vid,feature.shape)
        features[vid] = feature
    return features

if __name__ == '__main__':
    score_record = load_score(SCORE_PATH)
    features = load_feature(score_record, FEATURE_BASE)
    vids = list(score_record.keys())
    segment_info = {}
    for i in range(len(vids)):
        vid = vids[i]
        print('-'*20,i,vid,'-'*20)
        x = features[vid]
        y = score_record[vid]['scores_avg']
        vlength = math.ceil(len(x) / FRAME_RATE)  # 统一按照每两帧计一秒

        # segment
        max_weight = math.ceil(0.15 * vlength)
        K = np.dot(x, x.T)
        vmax = estimate_vmax(K)
        cps, scores = cpd_auto(K, vlength, vmax, lmin=1, lmax=max_weight)
        cps = np.append([0], np.append(cps, [len(x)], 0), 0)
        segment_info[vid] = cps
        print(cps)

    with open(SEGINFO_PATH, 'w') as file:
        json.dump(segment_info, file, cls=NpEncoder)

with open(SEGINFO_PATH, 'r') as file:
    seginfo = json.load(file)
with open(SCORE_PATH, 'r') as file:
    score_record = json.load(file)
vids = list(seginfo.keys())
for vid in vids:
    cps = seginfo[vid]
    fps = score_record[vid]['fps']
    frame_count = score_record[vid]['frame_count']
    vlength = frame_count / fps
    print(len(cps), vlength, vlength / len(cps))
print()
