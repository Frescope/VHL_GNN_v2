# 从oracle_summary中取尽可能少的query，包含全部的concept，生成part_label用于训练，在全部query上做测试

import os
import numpy as np
import math
import scipy.io
import logging
import json
import networkx as nx

QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
CONCEPT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'

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

def load_query_summary(query_sum_base):
    # 加载query-focused oracle summary
    summary = {}
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
    return summary

def load_concept(dict_path):
    concepts = []
    with open(dict_path, 'r') as f:
        for word in f.readlines():
            concepts.append(word.strip().split("'")[1])
    concepts.sort()
    return concepts

def query_find(summary):
    # 找出每个视频中，能够覆盖所有concept的最少query的集合
    Results = {'1':[],
               '2':[],
               '3':[],
               '4':[]}

    def cover_num(queries):
        covered = []
        for query in queries:
            covered += query.split('_')
        return len(set(covered))

    def backtrack(vid, chosen, covered, remains):
        # 回溯函数
        nonlocal UP
        if len(chosen) >= UP or UP <= 30:
            return
        if len(chosen) >= DN and len(covered) == C_NUM:
            Results[str(vid)].append(chosen)
            UP = len(chosen)  # 收缩query数量上限
            print(vid,len(chosen),len(Results[str(vid)]))
            return
        for i in range(len(remains)):
            chosen_next = chosen+remains[i:i+1]
            covered_next = covered | set(remains[i].split('_'))
            backtrack(vid, chosen_next, covered_next, remains[i+1:])
        return

    for i in range(1, 5):
        queries = list(summary[str(i)].keys())
        C_NUM = cover_num(queries)  # 当前视频中所有query包含的concept
        UP = 100  # query数量上限
        DN = int(C_NUM / 2)  # query数量下限
        backtrack(i, [], set([]), queries)
    return Results

def shots_count(queries, summary):
    # 计算所有query对应的shot总数，即label矩阵中的1数量
    num = 0
    for query in queries:
        num += len(summary[query])
        return num

def label_build(vid, queries, summary, concepts):
    video_lens = [2783, 3692, 2152, 3588]
    label = np.zeros((len(concepts), video_lens[i - 1]))  # 48*N
    for query in queries:
        hl_shots = summary[str(vid)][query]
        c1, c2 = query.split('_')
        ind1 = concepts.index(c1)
        ind2 = concepts.index(c2)
        label[ind1][hl_shots] += 1
        label[ind2][hl_shots] += 1
    return label

summary = load_query_summary(QUERY_SUM_BASE)
concepts = load_concept(CONCEPT_PATH)
# results = query_find(summary)
# with open(r'/data/linkang/VHL_GNN/utc/query_cover.json', 'w') as file:
#     json.dump(results,file)
# print('Query Covered !')

with open(r'/data/linkang/VHL_GNN/utc/query_cover.json', 'r') as file:
    results = json.load(file)

labels = {}
query_split = {}
for i in range(1, 5):
    # 找出query数量最少的组合
    min_num = 9999
    min_ind = 0
    for j in range(len(results[str(i)])):
        if len(results[str(i)][j]) < min_num:
            min_num = len(results[str(i)][j])
            min_ind = j
    # 找出这些组合中的正例最多的组合
    max_num = 0
    max_ind = 0
    for j in range(len(results[str(i)])):
        if len(results[str(i)][j]) == min_num:
            pos_num = shots_count(results[str(i)][j], summary[str(i)])
            if pos_num > max_num:
                max_num = pos_num
                max_ind = j
    train_queries = results[str(i)][max_ind]
    labels[str(i)] = label_build(i, train_queries, summary, concepts)
    remain_queries = list(set(summary[str(i)]) - set(train_queries))
    sp = int(len(remain_queries) / 2)
    valid_queries, test_queries = remain_queries[:sp], remain_queries[sp:]
    query_split[str(i)] = {
        'train_queries': train_queries,
        'valid_queries': valid_queries,
        'test_queries': test_queries
    }
with open(r'/data/linkang/VHL_GNN/utc/videotrans_label_part.json', 'w') as file:
    json.dump(labels,file,cls=NpEncoder)
with open(r'/data/linkang/VHL_GNN/utc/query_split.json', 'w') as file:
    json.dump(query_split,file,cls=NpEncoder)
print('Label Built !')