# 制作用于video_trans的标签，对于每个shot给出一组标记向量，向量中的每个元素对应一个concept，代表shot是否应当被选入与这个concept相关的summary

# 方案1：根据concept标签训练模型，再根据concept预测summary
# 最终目的是根据每个shot是否与一个concept相关，决定是否将这个shot选入包含这个concept的query-summary中
# 考虑到不同的query-summary中可能具有相同的concept但是标记的shot不同，因此需要验证根据concept标签选出的summary与真实summary的匹配程度
# 对于在更多query-summary中出现的shot，在对应的concept位置上给出更高的权重，预测时累计所有concept上的权重，选出总权重最高的shot进入summary

# 方案2：直接用summary标签训练模型，预测结果的每个分量直接对应一个query

import os
import numpy as np
import math
import scipy.io
import logging
import json
import networkx as nx

QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
CONCEPT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'

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

def label_build_scheme1(query_summary, concept_words):
    # 为每个video每个shot生成一个向量，每个分量按字典序对应一个concept
    video_lens = [2783, 3692, 2152, 3588]
    labels = {}
    for i in [1,2,3,4]:
        summary = query_summary[str(i)]
        label = np.zeros((len(concept_words), video_lens[i-1]))  # 48*N
        for query in summary:
            hl_shots = summary[query]
            c1, c2 = query.split('_')
            ind1 = concept_words.index(c1)
            ind2 = concept_words.index(c2)
            label[ind1][hl_shots] += 1
            label[ind2][hl_shots] += 1
        labels[str(i)] = label
    return labels

def label_build_scheme2(query_summary):
    # 每个shot的标签中，每个分量直接对应一个query，代表shot在query-summary中是否存在
    # 注意按照字典序排列
    video_lens = [2783, 3692, 2152, 3588]
    labels = {}
    for vid in [1,2,3,4]:
        queries = []
        for query in query_summary[str(vid)]:
            queries.append(query.split('_'))
        queries.sort(key=lambda x: (x[0], x[1]))
        label = np.zeros((len(queries), video_lens[vid-1]))
        for i in range(len(queries)):
            query = '_'.join(queries[i])
            hl_shots = query_summary[str(vid)][query]
            label[i][hl_shots] = 1
        labels[str(vid)] = label
    return labels

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

def label_check(query_summary, concept_words, labels):
    # 检查从labels中预测的summary与真实summary的匹配程度
    TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
    Tags = load_Tags(TAGS_PATH)
    for vid in [1,2,3,4]:
        PRE_values = []
        REC_values = []
        F1_values = []
        label = labels[str(vid)]
        summary = query_summary[str(vid)]
        hl_num = math.ceil(label.shape[-1] * 0.02)
        for query in summary:
            c1, c2 = query.split('_')
            ind1 = concept_words.index(c1)
            ind2 = concept_words.index(c2)
            scores = label[ind1] + label[ind2]
            pred = np.argsort(scores)[-hl_num:]
            pred.sort()
            gt = summary[query]
            # compute
            sim_mat = similarity_compute(Tags, int(vid), pred, gt)
            weight = shot_matching(sim_mat)
            precision = weight / len(pred)
            recall = weight / len(gt)
            f1 = 2 * precision * recall / (precision + recall)
            PRE_values.append(precision)
            REC_values.append(recall)
            F1_values.append(f1)
        PRE_values = np.array(PRE_values)
        REC_values = np.array(REC_values)
        F1_values = np.array(F1_values)
        print('Vid: %d, P: %.3f, R: %.3f, F: %.3f' %
              (vid, np.mean(PRE_values), np.mean(REC_values), np.mean(F1_values))
              )


if __name__ == '__main__':
    query_summary = load_query_summary(QUERY_SUM_BASE)
    concept_words = []
    with open(CONCEPT_PATH, 'r') as f:
        for word in f.readlines():
            concept_words.append(word.strip().split("'")[1])
    concept_words.sort()
    #
    # labels_s1 = label_build_scheme1(query_summary, concept_words)
    # labels_s2 = label_build_scheme2(query_summary)
    #
    # with open(r'/data/linkang/VHL_GNN/utc/videotrans_label_s1.json', 'w') as file:
    #     json.dump(labels_s1, file, cls=NpEncoder)
    # with open(r'/data/linkang/VHL_GNN/utc/videotrans_label_s2.json', 'w') as file:
    #     json.dump(labels_s2, file, cls=NpEncoder)
    # label_check(query_summary, concept_words, labels_s1)

    with open(r'/data/linkang/VHL_GNN/utc/videotrans_label_s1.json', 'r') as file:
        l1s = json.load(file)
    with open(r'/data/linkang/VHL_GNN/utc/videotrans_label_s2.json', 'r') as file:
        l2s = json.load(file)

    for vid in [1,2,3,4]:
        l1 = l1s[str(vid)]
        print('S1 Vid: ',vid)
        for i in range(len(l1)):
            count = 0
            for j in range(len(l1[i])):
                if l1[i][j] > 0: count += 1
            print(i, concept_words[i], count, '%.3f' % (count / len(l1[i])))
        print()

    for vid in [1,2,3,4]:
        l2 = l2s[str(vid)]
        print('S2 Vid: ',vid)
        queries = []
        for query in query_summary[str(vid)]:
            queries.append(query.split('_'))
        queries.sort(key=lambda x: (x[0], x[1]))
        for i in range(len(l2)):
            count = sum(l2[i])
            print(i, queries[i], count, '%.3f' % (count / len(l2[i])))
        print()
