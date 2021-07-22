# 根据concept tag从视频中获取对应片段的I3D特征（或resent特征）作为concept嵌入
# 每个视频都对应一组concept嵌入，验证和测试时都使用训练集的嵌入，防止引入测试信息

import numpy as np
import os
import json
import scipy.io
import h5py

DICT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
LABELS_PATH = r'/data/linkang/VHL_GNN/utc/concept_label.json'
VIDEO_FEAT_BASE = r'/data/linkang/VHL_GNN/utc/features/'
FEATURE_DIR = r'/data/linkang/VHL_GNN/utc/concept_video_embeddding/'

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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

def load_info(labels_path, dict_path):
    # 从Tags中加载每个视频中每个shot对应的concept标签
    with open(labels_path, 'r') as f:
        concept_labels = json.load(f)

    concepts = []
    with open(dict_path, 'r') as f:
        for word in f.readlines():
            concepts.append(word.strip().split("'")[1])
    concepts.sort()

    return concept_labels, concepts

def main():
    if not os.path.isdir(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)

    concept_labels, concepts = load_info(LABELS_PATH, DICT_PATH)

    for vid in range(1, 5):
        # 对每个视频生成一个目录
        feature_base = FEATURE_DIR + 'V0%d' % vid
        if not os.path.isdir(feature_base):
            os.makedirs(feature_base)

        # video_feat = np.load(VIDEO_FEAT_BASE + 'V%d_I3D.npy' % vid)
        f = h5py.File(VIDEO_FEAT_BASE + 'V%d_resnet_avg.h5' % vid, 'r')
        video_feat = f['feature'][()]

        concept_label = np.array(concept_labels[str(vid)])
        for i in range(len(concepts)):
            shot_relv = np.where(concept_label[:, i] > 0)[0]
            if len(shot_relv) == 0:
                print('\nEmpty: ', vid, i, concepts[i], '\n')
            embedding = video_feat[shot_relv]
            # feature_path = feature_base + '/' + concepts[i] + '_V%d_I3D.npy' % vid
            feature_path = feature_base + '/' + concepts[i] + '_V%d_RN.npy' % vid
            np.save(feature_path, embedding)
            print(vid, concepts[i], embedding.shape)

    return

if __name__ == '__main__':
    main()

