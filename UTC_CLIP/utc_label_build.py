# 用于改进版两阶段预测策略的特征提取

import json
import os
import scipy.io
import numpy as np
import logging

TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
CONCEPT_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/concept_label.json'
SUMMARY_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Global_Summaries/'
QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'

SHOTS_NUMS = [2783, 3692, 2152, 3588]

logging.basicConfig(level=logging.INFO)

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

# 验证concept_labels与Tags
def load(Tags_path, concept_label_path):
    # 从Tags中加载每个视频中每个shot对应的concept标签
    Tags = []
    Tags_raw = scipy.io.loadmat(Tags_path)
    Tags_tmp1 = Tags_raw['Tags'][0]
    logging.info('Tags: ')
    for i in range(4):
        Tags_tmp2 = Tags_tmp1[i][0]
        shot_labels = np.zeros((0, 48))
        for j in range(len(Tags_tmp2)):
            shot_label = Tags_tmp2[j][0][0].reshape((1, 48))
            shot_labels = np.vstack((shot_labels, shot_label))
        Tags.append(shot_labels)
        logging.info(str(i)+' '+str(shot_labels.shape))

    # load concept labels
    with open(concept_label_path, 'r') as file:
        concept_labels = json.load(file)

    return Tags, concept_labels

def check():
    Tags, concept_labels = load(TAGS_PATH, CONCEPT_LABEL_PATH)
    for i in range(len(Tags)):
        tags = Tags[i]
        labels = np.array(concept_labels[str(i + 1)])
        print(i, tags.shape, labels.shape)
        for j in range(len(tags)):
            if np.sum(abs(tags[j] - labels[j])) > 0:
                print(j)
                print(tags[j])
                print(labels[j])
                print()

# check()

# 提取generic summary label
def load_summary(summary_base):
    summary = {}
    for i in range(1, 5):
        summary_path = summary_base + 'P0%d/oracle.txt' % i
        with open(summary_path, 'r') as f:
            hl_shots = []
            for line in f.readlines():
                hl_shots.append(int(line.strip()) - 1)
            summary[str(i)] = hl_shots
    return summary

def label_build(summary):
    summary_labels = {}
    for i in range(4):
        hl_shots = summary[str(i + 1)]
        labels = np.zeros(SHOTS_NUMS[i])
        labels[hl_shots] = 1
        summary_labels[str(i + 1)] = labels
    return summary_labels

summary = load_summary(SUMMARY_BASE)
summary_labels = label_build(summary)
# with open(r'/data/linkang/VHL_GNN/utc/summary_label.json', 'w') as file:
#     json.dump(summary_labels, file, cls=NpEncoder)
# print('Done !')
#
# with open(r'/data/linkang/VHL_GNN/utc/summary_label.json', 'r') as file:
#     st = json.load(file)
# print()

def load_query_summary(query_sum_base):
    # 加载query-focused oracle summary
    summary = {}
    queries = {}  # 与label_s2中的query顺序一致
    for i in range(1,5):
        summary[str(i)] = {}
        summary_dir = query_sum_base + 'P0%d/' % i
        for root, dirs, files in os.walk(summary_dir):
            for file in files:
                concepts = file[:file.find("_oracle.txt")].split('_')  # 提取query单词，置换并排序
                concepts.sort()
                hl_shots = []
                with open(os.path.join(root,file),'r') as f:
                    for line in f.readlines():
                        hl_shots.append(int(line.strip())-1)
                summary[str(i)]['_'.join(concepts)] = hl_shots
        query_list = []
        for query in summary[str(i)]:
            query_list.append(query.split('_'))
        query_list.sort(key=lambda x: (x[0], x[1]))
        query_list = ['_'.join(x) for x in query_list]
        queries[str(i)] = query_list
    return queries, summary

def check2():
    queries, query_summary = load_query_summary(QUERY_SUM_BASE)
    for i in range(1, 5):
        s_labels = np.array(summary_labels[str(i)])
        s_shots = np.where(s_labels > 0)[0]
        print(i, len(s_labels))
        for query in query_summary[str(i)]:
            q_labels = np.array(query_summary[str(i)][query])
            q_shots = np.where(q_labels > 0)[0]
            union = set(list(s_shots) + list(q_shots))  # 取并集，观察重合度
            print(query, 'Union: %d, Sum: %d, Ratio: %.3f' %
                  (len(union), len(s_shots) + len(q_shots), len(union) / (len(s_shots) + len(q_shots))))
            # print('Generic: %d, %.3f, Query: %d, %.3f' %
            #       (len(s_shots), len(s_shots)/len(union), len(q_shots), len(q_shots)/len(union)))
        print()

check2()