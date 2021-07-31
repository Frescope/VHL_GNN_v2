# 使用CLIP特征的两阶段模型
# 将输入改为帧特征，将shot标签扩展为帧标签使用即，进行frame-level的训练
# 对于concept与summary分支，分别输出帧得分，得到总的帧得分后再聚合为shot得分

import os
import time
import numpy as np
import tensorflow as tf
import math
import json
import random
import logging
import argparse
import scipy.io
import h5py
import pickle
from trans2_2stages_clip_transformer import transformer
import networkx as nx

class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0',type=str)
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=6,type=int)
    parser.add_argument('--seq_len',default=25,type=int)  # clip数量
    parser.add_argument('--bc',default=20,type=int)
    parser.add_argument('--dropout',default='0.0',type=float)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--msd', default='clip', type=str)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--lr_noam', default=50e-6, type=float)
    parser.add_argument('--warmup', default=8500, type=int)
    parser.add_argument('--maxstep', default=100000, type=int)

    parser.add_argument('--qs_pr', default=0.1, type=float)  # query-summary positive ratio
    parser.add_argument('--concept_pr', default=0.5, type=float)

    parser.add_argument('--loss_concept_ratio', default=0.50, type=float)  # loss中来自concept_loss的比例
    parser.add_argument('--loss_reconst_ratio', default=0.00, type=float)  # loss中来自reconst_loss的比例
    parser.add_argument('--loss_diverse_ratio', default=0.00, type=float)  # loss中来自diverse_loss的比例

    parser.add_argument('--pred_concept_ratio', default=0.25, type=float)  # prediction中来自concept_logits的比例

    parser.add_argument('--global_ratio', default=0.1, type=float)  # 全局嵌入的抽样比例
    parser.add_argument('--global_mode', default='min', type=str)  # 全局嵌入的类型

    parser.add_argument('--repeat',default=3,type=int)
    parser.add_argument('--observe', default=0, type=int)
    parser.add_argument('--eval_epoch',default=10,type=int)
    parser.add_argument('--start', default='00', type=str)
    parser.add_argument('--end', default='', type=str)

hparams = Path()
parser = hparams.parser
hp = parser.parse_args()

if hp.server == 0:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
else:
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# global paras
D_VISUAL = 512
D_CONCEPT = 512
D_C_OUTPUT = 48  # concept, label_S1对应48，
D_S_OUTPUT = 45  # summary, label_S2对应45
CONCEPT_NUM = 48
MAX_F1 = 0.2
GRAD_THRESHOLD = 10.0  # gradient threshold2

LOAD_CKPT_MODEL = True
MIN_TRAIN_STEPS = 0
PRESTEPS = 0

if hp.server == 0:
    # path for JD server
    FEATURE_BASE = r'/public/data1/users/hulinkang/utc/features/'
    TAGS_PATH = r'/public/data1/users/hulinkang/utc/Tags.mat'
    CONCEPT_LABEL_PATH = r'/public/data1/users/hulinkang/utc/videotrans_label_s1.json'
    SUMMARY_LABEL_PATH = r'/public/data1/users/hulinkang/utc/videotrans_label_s2.json'
    QUERY_SUM_BASE = r'/public/data1/users/hulinkang/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/public/data1/users/hulinkang/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_PATH = r'/public/data1/users/hulinkang/utc/concept_clip.pkl'
    MODEL_SAVE_BASE = r'/public/data1/users/hulinkang/model_HL_utc_query/'
    CKPT_MODEL_PATH = r'/public/data1/users/hulinkang/model_HL_utc_query/video_trans/'
else:
    # path for USTC servers
    FEATURE_BASE = r'/data/linkang/VHL_GNN/utc/features/'
    TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
    CONCEPT_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_s1.json'
    SUMMARY_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_s2.json'
    QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_PATH = r'/data/linkang/VHL_GNN/utc/concept_clip.pkl'
    MODEL_SAVE_BASE = r'/data/linkang/model_HL_v4/'
    CKPT_MODEL_PATH = r'/data/linkang/model_HL_v4/utc_SA/'

logging.basicConfig(level=logging.INFO)

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
        query_list = ['_'.join(x) for x in query_list]
        queries[str(i)] = query_list
    return queries, summary

def load_feature_4fold(feature_base, concept_label_path, summary_label_path, Tags):
    # 注意label_s1对应的concept是按照字典序排列的，label_s2对应的query也是按照短语的字典序排列的
    # 加载特征 & 用于一阶段concept预测任务的标签 & 用于二阶段summary预测任务的标签
    with open(concept_label_path, 'r') as file:
        concept_labels = json.load(file)
    with open(summary_label_path, 'r') as file:
        summary_labels = json.load(file)

    data = {}
    for vid in range(1,5):
        data[str(vid)] = {}
        vlength = len(Tags[vid-1])
        # feature
        feature_path = feature_base + 'V%d_CLIP.npy' % vid
        feature = np.load(feature_path)
        data[str(vid)]['feature'] = feature

        # label for concept
        concept_label = np.array(concept_labels[str(vid)])[:,:vlength].T
        data[str(vid)]['concept_label'] = concept_label

        # label for summary
        summary_label = np.array(summary_labels[str(vid)])[:, :vlength].T
        data[str(vid)]['summary_label'] = summary_label

        logging.info('Vid: '+str(vid)+' Feature: '+str(feature.shape)+
                     ' Concept Label: '+str(concept_label.shape)+' Summary Label: '+str(summary_label.shape))
    return data

def load_concept(dict_path, concept_path):
    # 加载concept的字典、文本嵌入以及图像嵌入
    # load dict
    concepts = []
    with open(dict_path, 'r') as f:
        for word in f.readlines():
            concepts.append(word.strip().split("'")[1])
    concepts.sort()
    # load concept embedding
    with open(concept_path,'rb') as f:
        concept_embedding = pickle.load(f)
    return concepts, concept_embedding

def get_global_embeddings(data, ratio, mode):
    # 根据一定的比例从原视频的特征中随机采样一些，按照给定的模式拼接
    frame_num = len(data['concept_label']) * 5
    sample_num = min(math.ceil(frame_num * ratio), frame_num)
    if sample_num >= frame_num:
        embs_matrix = data['feature']
    else:
        samples = set()
        while len(samples) < sample_num:
            samples.add(np.random.randint(0, frame_num))
        embs_matrix = data['feature'][list(samples)]  # n*D
    if mode == 'mean':
        embs = np.mean(embs_matrix, axis=0)
    elif mode == 'max':
        embs = np.max(embs_matrix, axis=0)
    elif mode == 'min':
        embs = np.min(embs_matrix, axis=0)
    else:
        embs = np.mean(embs_matrix, axis=0)
    embs = embs.reshape((1, D_VISUAL))
    return embs

def train_scheme_build_2stages(data_train, concepts, query_summary, hp):
    # 用于两阶段预测的序列构建
    info_dict = {}
    for vid in data_train:
        label = data_train[vid]['concept_label']  # label for concept
        info_dict[vid] = {}
        for query in query_summary[vid]:
            info_dict[vid][query] = {}
            c1, c2 = query.split('_')
            ind1 = concepts.index(c1)
            ind2 = concepts.index(c2)
            c1_label = label[:, ind1]
            c2_label = label[:, ind2]
            c1_pos_list = list(np.where(c1_label > 0)[0])
            c1_neg_list = list(np.where(c1_label == 0)[0])
            c2_pos_list = list(np.where(c2_label > 0)[0])
            c2_neg_list = list(np.where(c2_label == 0)[0])
            qs_pos_list = query_summary[vid][query]
            random.shuffle(c1_pos_list)
            random.shuffle(c1_neg_list)
            random.shuffle(c2_pos_list)
            random.shuffle(c2_neg_list)
            random.shuffle(qs_pos_list)
            info_dict[vid][query]['c1_pos'] = c1_pos_list
            info_dict[vid][query]['c1_neg'] = c1_neg_list
            info_dict[vid][query]['c2_pos'] = c2_pos_list
            info_dict[vid][query]['c2_neg'] = c2_neg_list
            info_dict[vid][query]['qs_pos'] = qs_pos_list

    # 按照一定比例先取一些qs_pos节点，然后均分剩下的配额给两个concept
    train_scheme = []
    qs_num = math.ceil(hp.seq_len * hp.qs_pr)
    c1_part_len = math.ceil((hp.seq_len - qs_num) / 2)  # c1部分的序列长度
    c2_part_len = hp.seq_len - qs_num - c1_part_len
    cp_num = math.ceil((c1_part_len * hp.concept_pr))  # c1和c2部分序列中的正例数量
    for vid in data_train:
        vlength = len(data_train[vid]['concept_label'])
        for query in query_summary[vid]:
            c1_pos_list = info_dict[vid][query]['c1_pos']
            c1_neg_list = info_dict[vid][query]['c1_neg']
            c2_pos_list = info_dict[vid][query]['c2_pos']
            c2_neg_list = info_dict[vid][query]['c2_neg']
            k = math.ceil(max(len(c1_pos_list), len(c2_pos_list)) / cp_num)  # 取正例的循环次数，不足时从头循环
            qs_part = random.sample(info_dict[vid][query]['qs_pos'], qs_num)  # 随机取若干qs正例
            for i in range(k):  # 生成k个输入序列
                # 取c1相关的序列
                c1_pos_ind = i * cp_num % len(c1_pos_list)  # 从正例集合中取cp_nun个正例
                c1_neg_ind = i * (c1_part_len - cp_num) % len(c1_neg_list)  # c1_part的余下部分从负例集合中取
                c1_part = c1_pos_list[c1_pos_ind : c1_pos_ind + cp_num]
                c1_part += c1_neg_list[c1_neg_ind : c1_neg_ind + c1_part_len - cp_num]
                c1_part += c1_pos_list[0 : c1_part_len - len(c1_part)]  # 负例不足时做padding，一般不起作用
                # 取c2相关的序列
                c2_pos_ind = i * cp_num % len(c2_pos_list)  # 从正例集合中取cp_nun个正例
                c2_neg_ind = i * (c2_part_len - cp_num) % len(c2_neg_list)  # c2_part的余下部分从负例集合中取
                c2_part = c2_pos_list[c2_pos_ind: c2_pos_ind + cp_num]
                c2_part += c2_neg_list[c2_neg_ind: c2_neg_ind + c2_part_len - cp_num]
                c2_part += c2_pos_list[0: c2_part_len - len(c2_part)]  # 负例不足时做padding，一般不起作用
                clip_list = list(set(qs_part + c1_part + c2_part))
                while len(clip_list) < hp.seq_len:
                    pad_clip = random.randint(0, vlength - 1)
                    if pad_clip not in clip_list:
                        clip_list.append(pad_clip)
                clip_list.sort()
                train_scheme.append((vid, query, clip_list))
    random.shuffle(train_scheme)
    return train_scheme

def clip2frame(clip_list):
    frame_list = []
    for clip in clip_list:
        frame_list += list(range(clip * 5, (clip+1) * 5))
    return frame_list

def label_extend(label, d):
    # 将label标签扩展为frame标签
    label = np.reshape(label, (len(label), 1, d))
    label = np.tile(label, (1, 5, 1))
    label_new = np.reshape(label, (-1, d))
    return label_new

def get_batch_train_2stages(data_train, train_scheme, queries, concepts, step, hp):
    # 从train_scheme中获取gpu_num*bc个序列，每个长度seq_len，并返回每个clip的全局位置
    batch_num = hp.gpu_num * hp.bc
    features = []
    global_embs = []
    concept_labels = []
    summary_labels = []
    positions = []
    qc_indexes = []  # 用于在训练时确定每个序列对应的query与concept的分别的索引，用于合并两种logits
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(train_scheme)
        vid, query, clip_list = train_scheme[pos]
        frame_list = clip2frame(clip_list)
        features.append(data_train[vid]['feature'][frame_list])
        global_embs.append(get_global_embeddings(data_train[vid], hp.global_ratio, hp.global_mode))
        concept_labels.append(label_extend(data_train[vid]['concept_label'][clip_list], D_C_OUTPUT))
        summary_labels.append(label_extend(data_train[vid]['summary_label'][clip_list], D_S_OUTPUT))
        positions.append(frame_list)
        q_ind = queries[vid].index(query)
        c1, c2 = query.split('_')
        c1_ind = concepts.index(c1)
        c2_ind = concepts.index(c2)
        qc_indexes.append([q_ind, c1_ind, c2_ind])
    features = np.array(features)
    global_embs = np.array(global_embs)
    concept_labels = np.array(concept_labels)
    summary_labels = np.array(summary_labels)
    positions = np.array(positions)
    qc_indexes = np.array(qc_indexes)
    scores = np.ones((batch_num, hp.seq_len * 5 + 1))  # 多一个全局嵌入
    return features, global_embs, positions, scores, concept_labels, summary_labels, qc_indexes

def test_scheme_build(data_test, seq_len):
    # 依次输入测试集中所有clip，不足seqlen的要补足，在getbatch中补足不够一个batch的部分
    # (vid, seq_start, seq_end)形式
    test_scheme = []
    test_vids = []
    for vid in data_test:
        vlength = len(data_test[str(vid)]['concept_label'])
        seq_num = math.ceil(vlength / seq_len)
        for i in range(seq_num):
            test_scheme.append((vid, i * seq_len, min(vlength,(i+1) * seq_len)))
        test_vids.append((vid, vlength))
    return test_scheme, test_vids

def get_batch_test(data_test, test_scheme, step, hp):
    # 标记每个序列中的有效长度，并对不足一个batch的部分做padding
    # 不需要对序列水平上的padding做标记
    features = []
    global_embs = []
    positions = []
    scores = []
    batch_num = hp.gpu_num * hp.bc
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(test_scheme)
        vid, seq_start, seq_end = test_scheme[pos]  # clip的起止编号
        vlength = len(data_test[str(vid)]['concept_label'])
        padding_len = (hp.seq_len - (seq_end - seq_start)) * 5  # 需要填充的帧数量
        global_embs.append(get_global_embeddings(data_test[vid], 1.0, hp.global_mode))
        feature = data_test[str(vid)]['feature'][seq_start * 5 : seq_end * 5]
        position = np.array(list(range(seq_start * 5, seq_end * 5)))
        label = data_test[str(vid)]['concept_label'][seq_start:seq_end]
        score = np.ones(len(label) * 5 + 1)
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_VISUAL))
            position_pad = np.array([vlength * 5] * padding_len)
            score_pad = np.zeros(padding_len)
            feature = np.vstack((feature, feature_pad))
            position = np.hstack((position, position_pad))
            score = np.hstack((score, score_pad))
        features.append(feature)
        positions.append(position)
        scores.append(score)
    features = np.array(features)
    global_embs = np.array(global_embs)
    positions = np.array(positions)
    scores = np.array(scores)
    return features, global_embs, positions, scores

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def tower_loss_2stages(concept_logits, concept_labels, seq_logits, seq_labels, reconst_vecs, features, hp):
    # concept_logits & concept_labels: bc*seq_len*48
    # seq_logits & seq_labels: bc*seq_len
    # 对concept_loss，计算各个shot在所有concept上的NCE-Loss的均值
    # 对summary_loss，计算各个shot在其取样时对应的query上的NCE-Loss（为了防止负例比例过高）
    # 合并上述两种loss

    # for concept
    concept_logits = tf.clip_by_value(concept_logits, 1e-6, 0.999999)
    concept_labels_bin = tf.cast(tf.cast(concept_labels, dtype=tf.bool), dtype=tf.float32)
    concept_loss = - concept_labels_bin * tf.log(concept_logits) - (1 - concept_labels_bin) * tf.log(1 - concept_logits)
    concept_loss = tf.reduce_mean(concept_loss)

    # for summary
    seq_logits = tf.clip_by_value(seq_logits, 1e-6, 0.999999)
    summary_loss = - seq_labels * tf.log(seq_logits) - (1 - seq_labels) * tf.log(1 - seq_logits)
    summary_loss = tf.reduce_mean(summary_loss)

    # # for reconstruction
    # reconst_loss = tf.losses.mean_squared_error(reconst_vecs, features)
    # reconst_loss /= 1000
    #
    # # for diversity
    # diverse_labels = tf.reduce_sum(concept_labels, axis=2)  # 取所有concept正例的并集
    # labels_binary = tf.cast(tf.cast(diverse_labels, dtype=tf.bool), dtype=tf.float32)
    # labels_binary = tf.expand_dims(labels_binary, -1)  # bc*seq_len*1
    #
    # KeyVecs = reconst_vecs * labels_binary  # 遮蔽非关键片段，bc*seqlen*D
    # KeyVecs_T = tf.transpose(KeyVecs, perm=(0, 2, 1))  # bc*D*seqlen
    # Products = tf.matmul(KeyVecs, KeyVecs_T)  # 点积，bc*seqlen*seqlen
    #
    # Magnitude = tf.sqrt(tf.reduce_sum(tf.square(KeyVecs), axis=2, keep_dims=True))  # 求模，bc*seqlen*1
    # Magnitude_T = tf.transpose(Magnitude, perm=(0, 2, 1))  # bc*1*seqlen
    # Mag_product = tf.matmul(Magnitude, Magnitude_T)  # bc*seqlen*seqlen
    #
    # diverse_loss = tf.reduce_mean(Products / (Mag_product + 1e-8))

    # # total loss
    # r_c = hp.loss_concept_ratio
    # r_r = hp.loss_reconst_ratio
    # r_d = hp.loss_diverse_ratio
    # r_s = 1 - r_c - r_r - r_d
    # loss = concept_loss * r_c + reconst_loss * r_r + diverse_loss * r_d + summary_loss * r_s
    # return loss, [concept_loss, reconst_loss, diverse_loss, summary_loss]

    ratio_c = hp.loss_concept_ratio
    ratio_d = hp.loss_diverse_ratio

    # loss = concept_loss * ratio_c + diverse_loss * ratio_d + summary_loss * (1 - ratio_c - ratio_d)
    loss = concept_loss * ratio_c + summary_loss * (1 - ratio_c)
    return loss, [concept_loss, summary_loss]

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

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

def evaluation_2stages(concept_lists, summary_lists, query_summary, Tags, test_vids, concepts, queries):
    # 首先根据两组concept_logits选出一组候选集，然后从候选里根据summary_logits做最终预测
    c_logits = concept_lists[0]  # (bc*seq_len*5) * 48
    s_logits = summary_lists[0]  # (bc*seq_len*5) * 45
    for i in range(1, len(concept_lists)):
        c_logits = np.vstack((c_logits, concept_lists[i]))
        s_logits = np.vstack((s_logits, summary_lists[i]))
    c_logits = np.reshape(c_logits, (-1, 5, D_C_OUTPUT))
    c_logits = np.mean(c_logits, 1)  # 合并属于同一shot的帧得分
    s_logits = np.reshape(s_logits, (-1, 5, D_S_OUTPUT))
    s_logits = np.mean(s_logits, 1)

    pos = 0
    PRE_values = []
    REC_values = []
    F1_values = []
    ratio = hp.pred_concept_ratio
    for i in range(len(test_vids)):
        vid, vlength = test_vids[i]
        summary = query_summary[str(vid)]
        hl_num_s1 = math.ceil(vlength * 0.2)  # stage 1, 每组concept_logits中选20%进入候选集
        hl_num_s2 = math.ceil(vlength * 0.02)  # stage 2, 最终取2%作为summary
        c_predictions = c_logits[pos : pos + vlength]
        s_predictions = s_logits[pos : pos + vlength]
        pos += vlength
        for query in summary:
            shots_gt = summary[query]
            q_ind = queries[vid].index(query)
            c1, c2 = query.split('_')
            c1_ind = concepts.index(c1)
            c2_ind = concepts.index(c2)
            c1_logits = c_predictions[:, c1_ind]
            c2_logits = c_predictions[:, c2_ind]
            q_logits = s_predictions[:, q_ind]
            scores = (c1_logits + c2_logits) / 2 * ratio + q_logits * (1 - ratio)

            # find candidates
            c1_candidate = set(np.argsort(c1_logits)[-hl_num_s1 : ])
            c2_candidate = set(np.argsort(c2_logits)[-hl_num_s1 : ])
            candidate = list(c1_candidate | c2_candidate)
            candidate.sort()
            candidate = np.array(candidate).reshape((-1, 1))
            scores_indexes = scores[candidate].reshape((-1, 1))
            scores_indexes = np.hstack((scores_indexes, candidate))  # N*2，候选集及其原索引

            # make summary
            shots_pred = scores_indexes[scores_indexes[:, 0].argsort()]
            shots_pred = shots_pred[-hl_num_s2 : , 1].astype(int)
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

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def model_search(model_save_dir, observe):
    def takestep(name):
        return int(name.split('-')[0].split('S')[-1])
    # 找到要验证的模型名称
    model_to_restore = []
    for root,dirs,files in os.walk(model_save_dir):
        for file in files:
            if file.endswith('.meta'):
                model_name = file.split('.meta')[0]
                model_to_restore.append(os.path.join(root, model_name))
    model_to_restore = list(set(model_to_restore))
    # model_to_restore.sort(key=takestep)
    #
    # if observe == 0:
    #     # 只取最高F1的模型
    #     model_kfold = []
    #     f1s = []
    #     for name in model_to_restore:
    #         f1 = name.split('-')[-1]
    #         if f1.startswith('F'):
    #             f1s.append(float(f1.split('F')[-1]))
    #     if len(f1s) == 0:
    #         return []  # 没有合格的模型
    #     f1_max = np.array(f1s).max()
    #     for name in model_to_restore:
    #         f1 = name.split('-')[-1]
    #         if f1.startswith('F') and float(f1.split('F')[-1]) >= f1_max:
    #             model_kfold.append(name)
    #     model_to_restore = model_kfold

    return model_to_restore

def make_summary(concept_logits, summary_logits, summary_labels, qc_indexes, hp):
    # 按照concepts与queries中的顺序，根据concept与summary的logits决定最终的预测
    # concept_logits: bc*seq_len*48
    # summary_logits & lables: bc*seq_len*45
    # output & labels: bc*seq_len
    seq_logits = []  # 每个序列都只对应一个特定的query
    seq_labels = []
    for i in range(hp.bc):
        s_ind = qc_indexes[i][0]
        c1_ind = qc_indexes[i][1]
        c2_ind = qc_indexes[i][2]
        c1_logits = concept_logits[i : i + 1, :, c1_ind]  # 取第i个序列中c1对应的一行，保持前两维
        c2_logits = concept_logits[i : i + 1, :, c2_ind]
        c_logits = (c1_logits + c2_logits) / 2 * hp.pred_concept_ratio  # 需要加到对应位置上的concept_logits
        s_logits = summary_logits[i : i + 1, :, s_ind] * (1 - hp.pred_concept_ratio)
        s_labels = summary_labels[i : i + 1, :, s_ind]
        seq_logits.append(c_logits + s_logits)
        seq_labels.append(s_labels)
    seq_logits = tf.concat(seq_logits, axis=0)
    seq_labels = tf.concat(seq_labels, axis=0)
    return seq_logits, seq_labels

def run_testing(data_train, data_test, queries, query_summary, Tags, concepts, concept_embeeding, model_path):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        features_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len * 5, D_VISUAL))
        global_embs_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, 1, D_VISUAL))
        positions_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.seq_len * 5))
        scores_src_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len * 5 + CONCEPT_NUM + 1))
        concept_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len * 5, D_C_OUTPUT))
        summary_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len * 5, D_S_OUTPUT))
        qc_indexes_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, 3))  # 训练时给每个序列标记一个取样时的query与concept的索引
        concept_emb_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, CONCEPT_NUM, D_CONCEPT))
        dropout_holder = tf.placeholder(tf.float32, shape=())
        training_holder = tf.placeholder(tf.bool, shape=())

        # training operations
        lr = noam_scheme(hp.lr_noam, global_step, hp.warmup)
        opt_train = tf.train.AdamOptimizer(lr)

        # graph building
        tower_grads_train = []
        concept_logits_list = []
        summary_logits_list = []
        loss_list = []
        loss_ob_list = []
        for gpu_index in range(hp.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                features = features_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                global_embs = global_embs_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                positions = positions_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                scores_src = scores_src_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                concept_labels = concept_labels_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                summary_labels = summary_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                qc_indexes = qc_indexes_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                concept_emb = concept_emb_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]

                # 整合concept与summary的预测，形成最终预测
                concept_logits, summary_logits, reconst_vecs = transformer(features, positions, scores_src, concept_emb, global_embs,
                                     dropout_holder, training_holder, hp, D_C_OUTPUT, D_S_OUTPUT)
                concept_logits_list.append(concept_logits)
                summary_logits_list.append(summary_logits)

                seq_logits, seq_labels = make_summary(concept_logits, summary_logits, summary_labels, qc_indexes, hp)  # 训练时每个序列只针对一个query预测summary
                loss, loss_ob = tower_loss_2stages(concept_logits, concept_labels,
                                                   seq_logits, seq_labels,
                                                   reconst_vecs, features,
                                                   hp)
                varlist = tf.trainable_variables()  # 全部训练
                # grads_train = opt_train.compute_gradients(loss, varlist)
                # thresh = GRAD_THRESHOLD  # 梯度截断 防止爆炸
                # grads_train_cap = [(tf.clip_by_value(grad, -thresh, thresh), var) for grad, var in grads_train]
                # tower_grads_train.append(grads_train_cap)
                loss_list.append(loss)
                loss_ob_list += loss_ob
        train_op = tf.no_op()

        # session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        # load model
        saver_overall = tf.train.Saver(max_to_keep=100)
        if LOAD_CKPT_MODEL:
            logging.info(' Ckpt Model Restoring: ' + model_path)
            saver_overall.restore(sess, model_path)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        train_scheme = train_scheme_build_2stages(data_train, concepts, query_summary, hp)
        test_scheme, test_vids = test_scheme_build(data_test, hp.seq_len)
        epoch_step = math.ceil(len(train_scheme) / (hp.gpu_num * hp.bc))
        max_test_step = math.ceil(len(test_scheme) / (hp.gpu_num * hp.bc))

        # concept embedding processing
        concept_emb_b = []
        for c in concepts:
            concept_emb_b.append(concept_embeeding[c])
        concept_emb_b = np.array(concept_emb_b).reshape([1, CONCEPT_NUM, D_CONCEPT])
        concept_emb_b = np.tile(concept_emb_b, [hp.gpu_num * hp.bc, 1, 1])  # (bc*gpu_num)*48*d

        # begin training
        ob_loss = []
        ob_sub_loss = []
        timepoint = time.time()
        for step in range(hp.maxstep):
            features_b, global_embs_b, positions_b, scores_b, concept_labels_b, summary_labels_b, query_indexes_b = \
                get_batch_train_2stages(data_train, train_scheme, queries, concepts, step, hp)
            scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, CONCEPT_NUM))))  # encoder中开放所有concept节点
            observe = sess.run([train_op] + loss_list + concept_logits_list + summary_logits_list + loss_ob_list,
                               feed_dict={features_holder: features_b,
                                          global_embs_holder: global_embs_b,
                                          positions_holder: positions_b,
                                          scores_src_holder: scores_src_b,
                                          concept_labels_holder: concept_labels_b,
                                          summary_labels_holder: summary_labels_b,
                                          qc_indexes_holder: query_indexes_b,
                                          concept_emb_holder: concept_emb_b,
                                          dropout_holder: hp.dropout,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1 + hp.gpu_num])
            sub_loss_batch = observe[-4:]
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均
            ob_sub_loss.append(sub_loss_batch)

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == hp.maxstep:
                train_scheme = train_scheme_build_2stages(data_train, concepts, query_summary, hp)  # shuffle train scheme
                duration = time.time() - timepoint
                timepoint = time.time()
                loss_array = np.array(ob_loss)
                ob_loss.clear()
                sub_loss_array = np.array(ob_sub_loss)
                sub_loss_array = np.mean(sub_loss_array, axis=0)
                ob_sub_loss.clear()
                logging.info(' Step %d: %.3f sec' % (step, duration))
                logging.info(' Evaluate: ' + str(step) + ' Epoch: ' + str(epoch))
                logging.info(' Average Loss: ' + str(np.mean(loss_array)) + ' Min Loss: ' + str(
                    np.min(loss_array)) + ' Max Loss: ' + str(np.max(loss_array)))
                # logging.info('C_Loss: %.4f R_Loss: %.4f D_Loss: %.4f S_Loss: %.4f' %
                #              (sub_loss_array[0],sub_loss_array[1],sub_loss_array[2],sub_loss_array[3]))
                if not int(epoch) % hp.eval_epoch == 0:
                    continue  # 增大测试间隔
                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                concept_lists = []
                summary_lists = []
                for test_step in range(max_test_step):
                    features_b, global_embs_b, positions_b, scores_b = \
                        get_batch_test(data_test, test_scheme, test_step, hp)
                    scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, CONCEPT_NUM))))  # encoder中开放所有concept节点
                    temp_list = sess.run(concept_logits_list + summary_logits_list,
                                                             feed_dict={features_holder: features_b,
                                                                        global_embs_holder: global_embs_b,
                                                                        positions_holder: positions_b,
                                                                        scores_src_holder: scores_src_b,
                                                                        concept_emb_holder: concept_emb_b,
                                                                        dropout_holder: hp.dropout,
                                                                        training_holder: False})
                    for preds in temp_list[ : hp.gpu_num]:
                        concept_lists.append(preds.reshape((-1, D_C_OUTPUT)))
                    for preds in temp_list[hp.gpu_num : ]:
                        summary_lists.append(preds.reshape((-1, D_S_OUTPUT)))

                # p, r, f = evaluation(pred_scores, queries, query_summary, Tags, test_vids, concepts)
                p, r, f = evaluation_2stages(concept_lists, summary_lists, query_summary, Tags, test_vids, concepts, queries)
                logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f))
                return f
    return 0

def main(self):
    Tags = load_Tags(TAGS_PATH)
    data = load_feature_4fold(FEATURE_BASE, CONCEPT_LABEL_PATH, SUMMARY_LABEL_PATH, Tags)
    queries, query_summary = load_query_summary(QUERY_SUM_BASE)
    concepts, concept_embedding = load_concept(CONCEPT_DICT_PATH, CONCEPT_PATH)

    # evaluate all videos in turn
    model_scores = {}
    for kfold in range(4):
        # split data
        data_train = {}
        data_valid = {}
        data_test = {}
        data_train[str((kfold + 0) % 4 + 1)] = data[str((kfold + 0) % 4 + 1)]
        data_train[str((kfold + 1) % 4 + 1)] = data[str((kfold + 1) % 4 + 1)]
        data_valid[str((kfold + 2) % 4 + 1)] = data[str((kfold + 2) % 4 + 1)]
        data_test[str((kfold + 3) % 4 + 1)] = data[str((kfold + 3) % 4 + 1)]

        # info
        logging.info('*' * 20 + 'Settings' + '*' * 20)
        logging.info('K-fold: ' + str(kfold))
        logging.info('Train: %d, %d' % ((kfold + 0) % 4 + 1, (kfold + 1) % 4 + 1))
        logging.info('Valid: %d  Test: %d' % ((kfold + 2) % 4 + 1, (kfold + 3) % 4 + 1))
        logging.info('Model Base: ' + MODEL_SAVE_BASE + hp.msd + '_%d' % kfold)
        logging.info('WarmUp: ' + str(hp.warmup))
        logging.info('Noam LR: ' + str(hp.lr_noam))
        logging.info('Num Heads: ' + str(hp.num_heads))
        logging.info('Num Blocks: ' + str(hp.num_blocks))
        logging.info('Batchsize: ' + str(hp.bc))
        logging.info('Max Steps: ' + str(hp.maxstep))
        logging.info('Dropout Rate: ' + str(hp.dropout))
        logging.info('Sequence Length: ' + str(hp.seq_len))
        logging.info('Evaluation Epoch: ' + str(hp.eval_epoch))
        logging.info('Query Positive Ratio: ' + str(hp.qs_pr))
        logging.info('Concept Positive Ratio: ' + str(hp.concept_pr))
        logging.info('Loss Concept Ratio: ' + str(hp.loss_concept_ratio))
        logging.info('Loss Reconstruct Ratio: ' + str(hp.loss_reconst_ratio))
        logging.info('Loss Diverse Ratio: ' + str(hp.loss_diverse_ratio))
        logging.info('Pred Concept Ratio: ' + str(hp.pred_concept_ratio))
        logging.info('*' * 50)

        # repeat
        scores = []
        for i in range(hp.repeat):
            model_save_dir = MODEL_SAVE_BASE + hp.msd + '_%d_%d/' % (kfold, i) 
            models_to_restore = model_search(model_save_dir, observe=hp.observe)
            for i in range(len(models_to_restore)):
                logging.info('-' * 20 + str(i) + ': ' + models_to_restore[i].split('/')[-1] + '-' * 20)
                model_path = models_to_restore[i]
                f1 = run_testing(data_train, data_test, queries, query_summary, Tags, concepts, concept_embedding,
                                 model_path)
                scores.append(f1)
        model_scores[str((kfold + 3) % 4 + 1)] = scores
    scores_all = 0
    for vid in model_scores:
        scores = model_scores[vid]
        logging.info('Vid: %s, Mean: %.3f, Scores: %s' %
                     (vid, np.array(scores).mean(), str(scores)))
        scores_all += np.array(scores).mean()
    logging.info('Overall Results: %.3f' % (scores_all / 4))

if __name__ == '__main__':
    tf.app.run()
