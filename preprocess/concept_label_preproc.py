# 用于多任务学习的标签，来自于dense_per_shot_tags

import os
import numpy as np
import math
import json

CONCEPT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
TAG_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/'

CONCEPT_NUM = 48

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

def load_tags(tag_base):
    tags = {}
    for i in range(1, 5):
        tag_list = []
        tag_path = tag_base + 'P0%d/P0%d.txt' % (i, i)
        with open(tag_path, 'r') as f:
            for line in f.readlines():
                concepts = line.strip().split(',')
                tag_list.append(concepts)
        tags[str(i)] = tag_list
    return tags

def label_build(tags, concept_words):
    # 为每个视频的每个shot生成对应标签
    labels = {}
    for i in range(1, 5):
        tags_video = tags[str(i)]
        labels_video = []
        for tags_shot in tags_video:
            labels_shot = np.zeros(CONCEPT_NUM)
            tag_ids = [concept_words.index(tag) for tag in tags_shot]
            labels_shot[tag_ids] = 1
            labels_video.append(labels_shot)
        labels[str(i)] = np.array(labels_video)
    return labels

if __name__ == '__main__':
    concept_words = []
    with open(CONCEPT_PATH, 'r') as f:
        for word in f.readlines():
            concept_words.append(word.strip().split("'")[1])
    concept_words.sort()

    tags = load_tags(TAG_BASE)
    labels = label_build(tags, concept_words)
    for vid in labels:
        print(labels[vid].shape)

    with open(r'/data/linkang/VHL_GNN/utc/concept_label.json', 'w') as file:
        json.dump(labels, file, cls=NpEncoder)
    print('Done !')