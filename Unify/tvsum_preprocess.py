# TVSum预处理流程：取帧-取特征-分段-生成训练/测试标签
# 取帧按照2fps，使用之前已取好的帧以及对应的帧标签即可
# 取特征过程应支持多种特征：googlenet，resnet，clip等
# 使用kts算法按照特征做分段
# 按照分段结果，使用背包算法选出keyshot，生成标签

# 对TVSum进行聚类，每个视频抽象出几类，给每个帧赋予类标签

import json
import os
import random
import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import math
from tools.cpd_auto import cpd_auto, estimate_vmax
from tools.knapsack_iter import knapSack
from sklearn.cluster import KMeans

FEATURE_TYPE = 'googlenet'
# FEATURE_TYPE = 'resnet152'

FRAME_BASE = r'/data/linkang/tvsum50/frame_2fps/'
FEATURE_BASE = r'/data/linkang/tvsum50/feature_' + FEATURE_TYPE + '_2fps/'
SEGINFO_PATH = r'/data/linkang/tvsum50/segment_info.json'
SCORE_PATH = r'/data/linkang/VHL_GNN/tvsum_score_record.json'
LABEL_PATH = r'/data/linkang/tvsum50/score_record.json'
LABEL_CLUSTER_PATH = r'/data/linkang/tvsum50/score_record_cluster.json'

# FRAME_BASE = r'/public/data1/users/hulinkang/tvsum/frame/'
# FEATURE_BASE = r'/public/data1/users/hulinkang/tvsum/feature_' + FEATURE_TYPE + '_2fps/'
# SEGINFO_PATH = r'/public/data1/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_segment_info.json'
# SCORE_PATH = r'/public/data1/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_score_record.json'
# LABEL_PATH = r'/public/data1/users/hulinkang/tvsum/score_record.json'

FPS = 2
N_CLUSTERS = 4
N_CLUSTER_SAMPLE = 100  # 用于聚类的采样数（每个视频）

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

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

def knapSack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]

    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]


    best = K[n][W]

    amount = np.zeros(n);
    a = best;
    j = n;
    Y = W;

    while a > 0:
       while K[j][Y] == a:
           j = j - 1;

       j = j + 1;
       amount[j-1] = 1;
       Y = Y - wt[j-1];
       j = j - 1;
       a = K[j][Y];

    return amount

# 取帧
# 暂不需要

# 取特征
def featureExtract():

    def frame_cmp(name):
        return int(name.split('.jpg')[0])
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.isdir(FEATURE_BASE):
        os.makedirs(FEATURE_BASE)

    model = torch.hub.load('pytorch/vision:v0.6.0', FEATURE_TYPE, pretrained=True)

    # load images
    data = {}
    for root, dirs, files in os.walk(FRAME_BASE):
        for dir in dirs:
            vid = dir
            frame_names = os.listdir(os.path.join(root, vid))
            frame_names.sort(key=frame_cmp)
            input_tensor = []
            for name in frame_names:
                img = Image.open(os.path.join(root, vid, name))
                img_tensor = preprocess(img)
                img_tensor = img_tensor.unsqueeze(0)
                input_tensor.append(img_tensor)
            input_tensor = torch.cat(input_tensor, 0)
            data[vid] = input_tensor
            print(vid, input_tensor.size())

    # extract features
    # model = torch.hub.load('pytorch/vision:v0.6.0', FEATURE_TYPE, pretrained=True)
    model.eval()
    fmap_block = []
    input_block = []

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    model.fc.register_forward_hook(forward_hook)
    model.to('cuda')
    vids = list(data.keys())
    for i in range(len(vids)):
        vid = vids[i]
        print('-' * 20, i, vid, '-' * 20)
        input_batch = data[vid].to('cuda')
        with torch.no_grad():
            output = model(input_batch)
        features = input_block[0][0].squeeze()
        features = np.array(features.cpu())
        input_block.clear()

        # store features
        feature_path = FEATURE_BASE + vid + '_' + FEATURE_TYPE + '_2fps.npy'
        np.save(feature_path, features)
        print(data[vid].size(), features.shape)
    print(FEATURE_TYPE + 'Features Extracted !')

# KTS分段
def segmentSplit():
    # load data
    # load info
    with open(SCORE_PATH, 'r') as file:
        score_record = json.load(file)
    # load feature
    vids = list(score_record.keys())
    features = {}
    for vid in vids:
        feature_path = FEATURE_BASE + vid + '_' + FEATURE_TYPE + '_2fps.npy'
        feature = np.load(feature_path)
        print(vid, feature.shape)
        features[vid] = feature

    # generate segment
    segment_info = {}
    for i in range(len(vids)):
        vid = vids[i]
        print('-'*20,i,vid,'-'*20)
        x = features[vid]
        vlength = math.ceil(len(x) / FPS)  # 统一按照每两帧计一秒

        # segment
        max_weight = math.ceil(0.15 * vlength)
        K = np.dot(x, x.T)
        vmax = estimate_vmax(K)
        cps, scores = cpd_auto(K, vlength, vmax, lmin=1, lmax=max_weight)
        cps = np.append([0], np.append(cps, [len(x)], 0), 0)
        segment_info[vid] = cps
        print(vid, len(x), len(cps))

    with open(SEGINFO_PATH, 'w') as file:
        json.dump(segment_info, file, cls=NpEncoder)
    print('Segment Split !')

# 生成标签
def labelGenerate():
    # 暂行：
    # 使用所有评分者的平均分作为唯一得分，基于此得分计算keyshot-label，同时用于训练和测试

    def frame2shot(vid, segment_info, scores):
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

    with open(SCORE_PATH, 'r') as file:
        score_record = json.load(file)
    with open(SEGINFO_PATH, 'r') as file:
        segment_info = json.load(file)

    vids = list(score_record.keys())
    for vid in vids:
        scores = np.array(score_record[vid]['scores'])
        scores_avg = scores.mean(axis=0,keepdims=True)  # 1*vlength
        keyshot_labels = frame2shot(vid, segment_info, scores)
        score_record[vid]['keyshot_labels'] = keyshot_labels
        keyshot_label_uni = frame2shot(vid, segment_info, scores_avg)
        score_record[vid]['keyshot_label_uni'] = keyshot_label_uni
        print(vid, scores_avg.shape, sum(keyshot_label_uni) / len(keyshot_label_uni))

    with open(LABEL_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)
    print('Keyshot Labels Generated !')

# 聚类
def clustering():
    # 对视频的每一帧做聚类
    # load info
    with open(LABEL_PATH, 'r') as file:
        score_record = json.load(file)
    # load feature
    vids = list(score_record.keys())
    features = {}
    for vid in vids:
        feature_path = FEATURE_BASE + vid + '_' + FEATURE_TYPE + '_2fps.npy'
        feature = np.load(feature_path)
        features[vid] = feature
    # sample features & cluster
    X_sample = []
    for vid in vids:
        X_vid = features[vid]
        step = int(len(X_vid) / N_CLUSTER_SAMPLE)  # 均匀地从视频中抽取一些帧特征
        pos = 0
        while pos < len(X_vid):
            X_sample.append(X_vid[pos])
            pos += step
    X_sample = np.array(X_sample)
    cluster = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(X_sample)
    # generate cluster labels
    for vid in vids:
        X_vid = features[vid]
        label = cluster.predict(X_vid)
        score_record[vid]['cluster_label'] = label
        print(vid, X_vid.shape, label.shape)

    with open(LABEL_CLUSTER_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)
    print('Cluster Labels Generated !')

if __name__ == '__main__':
    # featureExtract()
    # segmentSplit()
    # labelGenerate()
    clustering()

