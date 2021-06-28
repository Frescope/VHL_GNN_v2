# 分析每个query对应的标签长度与视频总长的关系
import os
import time
import numpy as np
import math
import json
import random
import logging
import argparse
import scipy.io
import h5py
import pickle
import networkx as nx

TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_s1.json'
CONCEPT_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/concept_label.json'
CONCEPT_DICT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'

def load_Tags(Tags_path):
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
    return Tags

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
        queries[str(i)] = query_list
    return queries, summary

def shot_matching(sim_mat):
    # 根据相似度矩阵计算二分图的最大匹配后的总权重
    shot_num1, shot_num2 = sim_mat.shape
    G = nx.complete_bipartite_graph(shot_num1,shot_num2)
    left, right = nx.bipartite.sets(G)  # 节点编号从left开始计数
    left = list(left)
    right = list(right)
    for i in range(shot_num1):
        for j in range(shot_num2):
            G[left[i]][right[j]]['weight'] = sim_mat[i,j]
    edge_set = nx.algorithms.matching.max_weight_matching(G,maxcardinality=False,weight='weight')

    # 计算总权重
    weight_sum = 0
    for edge in list(edge_set):
        weight_sum += G[edge[0]][edge[1]]['weight']
    return weight_sum

def similarity_compute(Tags,vid,shot_seq1,shot_seq2):
    # 计算两个shot序列之间的相似度，返回相似度矩阵
    # 注意，返回的矩阵的行和列分别对应序列1和序列2中的顺序
    def concept_IOU(shot_i, shot_j):
        # 计算intersection-over-union
        intersection = shot_i * shot_j
        union = (shot_i + shot_j).astype('bool').astype('int')
        return np.sum(intersection) / np.sum(union)

    vTags = Tags[vid-1]
    shot_num1 = len(shot_seq1)
    shot_num2 = len(shot_seq2)
    sim_mat = np.zeros((shot_num1,shot_num2))
    for i in range(shot_num1):
        for j in range(shot_num2):
            sim_mat[i][j] = concept_IOU(vTags[shot_seq1[i]],vTags[shot_seq2[j]])
    return sim_mat

def evaluation(labels, query_summary, Tags, concepts):
    # 从每个视频的预测结果中，根据query中包含的concept选出相关程度最高的一组shot，匹配后计算f1，求所有query的平均结果
    PRE_values = []
    REC_values = []
    F1_values = []
    for vid in labels:
        vlength = len(labels[vid])
        summary = query_summary[vid]
        # hl_num = math.ceil(vlength * 0.02)
        predictions = labels[vid]
        for query in summary:
            shots_gt = summary[query]
            c1, c2 = query.split('_')

            # for s1
            ind1 = concepts.index(c1)
            ind2 = concepts.index(c2)
            scores = (predictions[:, ind1] + predictions[:, ind2]).reshape((-1))

            hl_num = len(shots_gt)
            shots_pred = np.argsort(scores)[-hl_num:]
            shots_pred.sort()
            # compute
            sim_mat = similarity_compute(Tags, int(vid), shots_pred, shots_gt)
            weight = shot_matching(sim_mat)
            precision = weight / len(shots_pred)
            recall = weight / len(shots_gt)
            f1 = 2 * precision * recall / (precision + recall)
            PRE_values.append(precision)
            REC_values.append(recall)
            F1_values.append(f1)
    PRE_values = np.array(PRE_values)
    REC_values = np.array(REC_values)
    F1_values = np.array(F1_values)
    return np.mean(PRE_values), np.mean(REC_values), np.mean(F1_values)

def main():
    Tags = load_Tags(TAGS_PATH)
    queries, query_summary = load_query_summary(QUERY_SUM_BASE)

    concepts = []
    with open(CONCEPT_DICT_PATH, 'r') as f:
        for word in f.readlines():
            concepts.append(word.strip().split("'")[1])
    concepts.sort()
    with open(LABEL_PATH, 'r') as file:
        labels_tmp = json.load(file)
    with open(CONCEPT_LABEL_PATH, 'r') as file:
        concept_labels_tmp = json.load(file)
    labels = {}
    concept_labels = {}
    for vid in labels_tmp:
        vlength = len(Tags[int(vid) - 1])
        labels[vid] = np.array(labels_tmp[vid])[:,:vlength].T
        concept_labels[vid] = np.array(concept_labels_tmp[vid])

    # 检验候选集
    for vid in range(4):
        summary = query_summary[str(vid+1)]
        predictions = labels[str(vid+1)]
        for query in summary:
            shots_gt = set(summary[query])
            c1, c2 = query.split('_')
            ind1 = concepts.index(c1)
            ind2 = concepts.index(c2)
            l1 = predictions[:, ind1]
            l2 = predictions[:, ind2]
            l = (l1 + l2).astype('bool').astype('int')
            candidate = set(np.where(l > 0)[0])
            print(vid, query, len(candidate), len(shots_gt), len(candidate - shots_gt))

    # concept_label与summary_label的差异
    for vid in range(4):
        vlength = len(Tags[vid])
        summary = query_summary[str(vid+1)]
        predictions = Tags[vid]
        for query in summary:
            shots_gt = summary[query]
            c1, c2 = query.split('_')

            # for s1
            ind1 = concepts.index(c1)
            ind2 = concepts.index(c2)
            l1 = predictions[:, ind1]
            l2 = predictions[:, ind2]
            l = l1 * l2
            hl_num = len(shots_gt)
            print(vid, query, int(np.sum(l1)), int(np.sum(l2)), int(np.sum(l)), hl_num)

    # f1上限
    p, r, f = evaluation(labels, query_summary, Tags, concepts)
    print(p, r, f)

    # gt占视频总长的比例
    for vid in query_summary:
        vlength = len(Tags[int(vid) - 1])
        gt_length = []
        for query in query_summary[vid]:
            shots_gt = query_summary[vid][query]
            gt_length.append(len(shots_gt))
        gt_length = np.array(gt_length)
        print(vid, vlength)
        print('\t\tMIN: ',np.min(gt_length), np.min(gt_length) / vlength)
        print('\t\tMAX: ', np.max(gt_length), np.max(gt_length) / vlength)
        print('\t\tMEAN:', int(np.mean(gt_length)), np.mean(gt_length) / vlength)
        print()

    # query重叠情况
    vqs = []
    for vid in query_summary:
        video_queries = set(query_summary[vid].keys())
        vqs.append(video_queries)
    for i in range(len(vqs)):
        for j in range(i + 1, len(vqs)):
            common = vqs[i] & vqs[j]
            print(i, j, len(vqs[i]), len(vqs[j]), len(common))




if __name__ == '__main__':
    main()