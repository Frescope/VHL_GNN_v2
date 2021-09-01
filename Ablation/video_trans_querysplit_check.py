# 基于seg_mem，在训练、验证与测试时均使用部分query

import os
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import math
import json
import random
import logging
import argparse

import scipy.io
import pickle
from transformer_seg_mem import transformer
import networkx as nx

class Path:
    parser = argparse.ArgumentParser()
    # 显卡，服务器与存储
    parser.add_argument('--gpu', default='0',type=str)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--msd', default='video_trans', type=str)

    # 训练参数
    parser.add_argument('--bc',default=20,type=int)
    parser.add_argument('--dropout',default='0.1',type=float)
    parser.add_argument('--lr_noam', default=1e-5, type=float)
    parser.add_argument('--warmup', default=8500, type=int)
    parser.add_argument('--maxstep', default=100000, type=int)
    parser.add_argument('--repeat', default=3, type=int)
    parser.add_argument('--observe', default=0, type=int)
    parser.add_argument('--eval_epoch', default=10, type=int)
    parser.add_argument('--start', default='00', type=str)
    parser.add_argument('--end', default='', type=str)
    parser.add_argument('--protection', default=1000, type=int)  # 不检查步数太小的模型

    # Encoder结构参数
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=6,type=int)

    # 序列参数，长度与正样本比例
    parser.add_argument('--seq_len',default=25,type=int)  # 视频序列的长度
    parser.add_argument('--qs_pr', default=0.1, type=float)  # query-summary positive ratio
    parser.add_argument('--concept_pr', default=0.5, type=float)

    # segment-embedding参数
    parser.add_argument('--segment_num', default=75, type=int)  # segment节点数量
    parser.add_argument('--segment_mode', default='min', type=str)  # segment-embedding的聚合方式

    # memory参数
    parser.add_argument('--memory_num', default=60, type=int)  # memory节点数量
    parser.add_argument('--memory_dimension', default=1024, type=int)  # memory节点的维度
    parser.add_argument('--memory_init', default='random', type=str)  # random, text

    # loss参数
    parser.add_argument('--loss_s1_ratio', default=0.80, type=float)  # pred损失比例
    parser.add_argument('--mem_div', default=0.10, type=float)  # memory_diversity损失比例
    parser.add_argument('--shots_div', default=0.10, type=float)  # shots_diversity损失比例
    parser.add_argument('--shots_div_ratio', default=0.20, type=float)  # shots_diversity中挑选出的片段比例

hparams = Path()
parser = hparams.parser
hp = parser.parse_args()

if hp.server != 0:  # 对JD server不能指定gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu

if hp.server != 1:  # 更高版本tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)


# global paras
D_FEATURE = 1024  # for I3D
D_TXT_EMB = 300
D_C_OUTPUT = 48  # concept, label_S1对应48，
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
    S1_LABEL_PATH = r'/public/data1/users/hulinkang/utc/videotrans_label_part.json'
    S2_LABEL_PATH = r'/public/data1/users/hulinkang/utc/videotrans_label_s2.json'
    SUMMARY_LABEL_PATH = r'/public/data1/users/hulinkang/utc/summary_label.json'
    QUERY_SUM_BASE = r'/public/data1/users/hulinkang/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/public/data1/users/hulinkang/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'/public/data1/users/hulinkang/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/public/data1/users/hulinkang/utc/concept_embeddding/'
    MODEL_SAVE_BASE = r'/public/data1/users/hulinkang/model_HL_utc_query/'
    CKPT_MODEL_PATH = r'/public/data1/users/hulinkang/model_HL_utc_query/video_trans/'
    QUERY_SPLIT_PATH = r'/public/data1/users/hulinkang/utc/query_split.json'
elif hp.server == 1:
    # path for USTC servers
    FEATURE_BASE = r'/data/linkang/VHL_GNN/utc/features/'
    TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
    S1_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_part.json'
    S2_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_s2.json'
    SUMMARY_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/summary_label.json'
    QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'/data/linkang/VHL_GNN/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/data/linkang/VHL_GNN/utc/concept_embeddding/'
    MODEL_SAVE_BASE = r'/data/linkang/model_HL_v4/'
    CKPT_MODEL_PATH = r'/data/linkang/model_HL_v4/utc_SA/'
    QUERY_SPLIT_PATH = r'/data/linkang/VHL_GNN/utc/query_split.json'
else:
    # path for JD A100 Server, make sure that VHL_GNN_V2 and utc folder are under the same directory
    FEATURE_BASE = r'../utc/features/'
    TAGS_PATH = r'../utc/Tags.mat'
    S1_LABEL_PATH = r'../utc/videotrans_label_s1.json'
    S2_LABEL_PATH = r'../utc/videotrans_label_s2.json'
    SUMMARY_LABEL_PATH = r'../utc/summary_label.json'
    QUERY_SUM_BASE = r'../utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'../utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'../utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'../utc/concept_embeddding/'
    MODEL_SAVE_BASE = r'/home/models/'
    CKPT_MODEL_PATH = r'/home/models/'

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

def load_concept(dict_path, txt_emb_path, img_emb_dir):
    # 加载concept的字典、文本嵌入以及图像嵌入
    # load dict
    concepts = []
    with open(dict_path, 'r') as f:
        for word in f.readlines():
            concepts.append(word.strip().split("'")[1])
    concepts.sort()
    # load text embedding
    with open(txt_emb_path,'rb') as f:
        txt_embedding = pickle.load(f)
    # load img embedding
    img_embedding = {}
    for root, dirs, files in os.walk(img_emb_dir):
        for file in files:
            concept = file.split('_')[0]
            embedding = np.load(os.path.join(root, file))
            img_embedding[concept] = np.mean(embedding, axis=0)
    # embedding integration
    concept_embedding = {}
    concept_transfer = {"Cupglass": "Glass",
                        "Musicalinstrument": "Instrument",
                        "Petsanimal": "Animal"}
    for key in img_embedding:
        concept_embedding[key] = {}
        txt_key = img_key = key
        if txt_key in concept_transfer:
            txt_key = concept_transfer[txt_key]
        concept_embedding[key]['txt'] = txt_embedding[txt_key]
        concept_embedding[key]['img'] = img_embedding[img_key]
    return concepts, concept_embedding

def load_feature_4fold(feature_base, s1_label_path, Tags):
    with open(s1_label_path, 'r') as file:
        s1_labels = json.load(file)

    data = {}
    for vid in range(1,5):
        data[str(vid)] = {}
        vlength = len(Tags[vid-1])
        # feature
        feature_path = feature_base + 'V%d_I3D.npy' % vid
        feature = np.load(feature_path)
        data[str(vid)]['feature'] = feature

        # label for s1
        s1_label = np.array(s1_labels[str(vid)])[:,:vlength].T
        data[str(vid)]['s1_label'] = s1_label

        logging.info('Vid: '+str(vid)+
                     ' Feature: '+str(feature.shape)+
                     ' S1 Label: '+str(s1_label.shape)
                     )
    return data

def segment_embedding_build(data, hp):
    # 根据预设的segment参数生成每个视频的segment_embeddings
    segnum = hp.segment_num
    mode = hp.segment_mode
    segment_dict = {}
    for vid in data:
        segment_dict[vid] = {}
        vlength = len(data[vid]['feature'])
        seglength = math.floor(vlength / segnum)  # 每个segment覆盖的范围
        segments = []
        poses = []
        for i in range(segnum):
            start = seglength * i
            end = min(vlength, seglength * (i + 1))
            mid = math.ceil((start + end) / 2)
            poses.append(mid)
            embs = data[vid]['feature'][start : end]
            if mode == 'mean':
                embs = np.mean(embs, axis=0)
            elif mode == 'max':
                embs = np.max(embs, axis=0)
            elif mode == 'min':
                embs = np.min(embs, axis=0)
            else:
                embs = np.mean(embs, axis=0)
            segments.append(embs)
        poses = np.array(poses)
        segments = np.array(segments)
        segment_dict[vid]['segment_emb'] = segments
        segment_dict[vid]['segment_pos'] = poses
    return segment_dict

def train_scheme_build(data_train, concepts, query_summary, hp):
    # 用于两阶段预测的序列构建
    info_dict = {}
    for vid in data_train:
        label = data_train[vid]['s1_label']  # label for concept
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
        vlength = len(data_train[vid]['s1_label'])
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

def get_batch_train(data_train, segment_dict, train_scheme, step, hp):
    # 从train_scheme中获取gpu_num*bc个序列，每个长度seq_len，并返回每个clip的全局位置
    batch_num = hp.gpu_num * hp.bc
    features = []
    s1_labels = []
    positions = []
    segment_embs = []
    segment_poses = []
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(train_scheme)
        vid, query, clip_list = train_scheme[pos]
        features.append(data_train[vid]['feature'][clip_list])
        s1_labels.append(data_train[vid]['s1_label'][clip_list])
        positions.append(clip_list)
        segment_embs.append(segment_dict[vid]['segment_emb'])
        segment_poses.append(segment_dict[vid]['segment_pos'])
    features = np.array(features)
    s1_labels = np.array(s1_labels)
    positions = np.array(positions)
    segment_embs = np.array(segment_embs)
    segment_poses = np.array(segment_poses)
    scores = np.ones((batch_num, hp.seq_len))
    return features, positions, segment_embs, segment_poses, scores, s1_labels

def test_scheme_build(data_test, seq_len):
    # 依次输入测试集中所有clip，不足seqlen的要补足，在getbatch中补足不够一个batch的部分
    # (vid, seq_start, seq_end)形式
    test_scheme = []
    test_vids = []
    for vid in data_test:
        vlength = len(data_test[str(vid)]['s1_label'])
        seq_num = math.ceil(vlength / seq_len)
        for i in range(seq_num):
            test_scheme.append((vid, i * seq_len, min(vlength,(i+1) * seq_len)))
        test_vids.append((vid, vlength))
    return test_scheme, test_vids

def get_batch_test(data_test, segment_dict, test_scheme, step, hp):
    # 标记每个序列中的有效长度，并对不足一个batch的部分做padding
    # 不需要对序列水平上的padding做标记
    features = []
    positions = []
    segment_embs = []
    segment_poses = []
    scores = []
    batch_num = hp.gpu_num * hp.bc
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(test_scheme)
        vid, seq_start, seq_end = test_scheme[pos]
        vlength = len(data_test[str(vid)]['s1_label'])
        padding_len = hp.seq_len - (seq_end - seq_start)
        feature = data_test[str(vid)]['feature'][seq_start:seq_end]
        position = np.array(list(range(seq_start, seq_end)))
        label = data_test[str(vid)]['s1_label'][seq_start:seq_end]
        score = np.ones(len(label))
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_FEATURE))
            position_pad = np.array([vlength] * padding_len)
            score_pad = np.zeros(padding_len)
            feature = np.vstack((feature, feature_pad))
            position = np.hstack((position, position_pad))
            score = np.hstack((score, score_pad))
        features.append(feature)
        positions.append(position)
        scores.append(score)
        segment_embs.append(segment_dict[vid]['segment_emb'])
        segment_poses.append(segment_dict[vid]['segment_pos'])
    features = np.array(features)
    positions = np.array(positions)
    segment_embs = np.array(segment_embs)
    segment_poses = np.array(segment_poses)
    scores = np.array(scores)
    return features, positions, segment_embs, segment_poses, scores

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

def tower_loss_diverse(pred_s1_logits, pred_s1_labels, shots_output, memroy_output, hp):
    # shots_output: bc*seq_len*D
    # memory_output: bc*memory_num*D
    # pred_s1_logits & pred_s1_labels: bc*seq_len*48
    # 对s1_loss，计算各个shot在s1中所有concept相关预测上的NCE-Loss的均值
    # 计算shots与memory的diversity-loss，s1_loss加权求和

    def diversity_loss(Vecs):
        # 计算一组输入节点特征之间的多样性损失
        vecs_len = Vecs.get_shape().as_list()[1]
        Vecs_T = tf.transpose(Vecs, perm=(0, 2, 1))  # bc*D*vecslen
        Products = tf.matmul(Vecs, Vecs_T)  # 点积，bc*vecslen*vecslen

        Magnitude = tf.sqrt(tf.reduce_sum(tf.square(Vecs), axis=2, keep_dims=True))  # 求模，bc*vecslen*1
        Magnitude_T = tf.transpose(Magnitude, perm=(0, 2, 1))  # bc*1*vecslen
        Mag_product = tf.matmul(Magnitude, Magnitude_T)  # bc*vecslen*vecslen

        loss = tf.reduce_sum(Products / (Mag_product + 1e-8), [1, 2])
        obs1 = Products / (Mag_product + 1e-8)
        loss = loss - vecs_len * 1 / (1 + 1e-8)  #减去每个序列中节点自身相乘得到的对角线上的1
        obs2 = loss
        loss = tf.reduce_mean(loss / (vecs_len - 1) / vecs_len)
        return loss, [obs1, obs2]

    # for s1，与分解到concept的query-summary label
    pred_s1_logits = tf.transpose(pred_s1_logits, perm=(0, 2, 1))  # bc*48*seq_len
    pred_s1_logits = tf.reshape(pred_s1_logits, shape=(-1, hp.seq_len))  # (bc*48)*seq_len
    s1_labels_flat = tf.transpose(pred_s1_labels, perm=(0, 2, 1))
    s1_labels_flat = tf.reshape(s1_labels_flat, shape=(-1, hp.seq_len))
    s1_labels_bin = tf.cast(tf.cast(s1_labels_flat, dtype=tf.bool), dtype=tf.float32)  # 转化为0-1形式，浮点数

    s1_nce_pos = tf.reduce_sum(tf.exp(s1_labels_bin * pred_s1_logits), axis=1)  # 分子
    s1_nce_pos -= tf.reduce_sum((1 - s1_labels_bin), axis=1)  # 减去负例（为零）取e后的值（为1）
    s1_nce_all = tf.reduce_sum(tf.exp(pred_s1_logits), axis=1)  # 分母
    s1_nce_loss = -tf.log((s1_nce_pos / s1_nce_all) + 1e-5)
    s1_loss = tf.reduce_mean(s1_nce_loss)

    # for shots diversity
    if hp.shots_div >= 0.01:
        top_50 = tf.nn.top_k(pred_s1_logits, int(hp.seq_len * hp.shots_div_ratio))
        top_indices = top_50.indices  # 没行（序列）前50%的索引
        KeyVecs = []  # 得分较高的shot对应的重建向量
        for i in range(top_indices.get_shape().as_list()[0]):  # i为展开后的pred第一维坐标
            seq_pos = int(i / CONCEPT_NUM)  # 对应的输出特征中的序列位置
            KeyVecs.append(tf.gather(shots_output[seq_pos:seq_pos+1], top_indices[i], axis=1))
        KeyVecs = tf.concat(KeyVecs, axis=0)
        shots_diverse_loss, _ = diversity_loss(KeyVecs)
    else:
        shots_diverse_loss = tf.convert_to_tensor(np.zeros(1), dtype=tf.float32)

    # for memory diversity
    if hp.mem_div >= 0.05:
        Vecs = memroy_output  # 全部节点都应当不相似
        mem_diverse_loss, obs = diversity_loss(Vecs)
    else:
        mem_diverse_loss = tf.convert_to_tensor(np.zeros(1), dtype=tf.float32)

    loss = s1_loss * hp.loss_s1_ratio  + \
           shots_diverse_loss * hp.shots_div + \
           mem_diverse_loss * hp.mem_div
    return loss, [s1_loss, shots_diverse_loss, mem_diverse_loss]

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
        return np.sum(intersection) / (np.sum(union) + 1e-9)

    vTags = Tags[vid-1]
    shot_num1 = len(shot_seq1)
    shot_num2 = len(shot_seq2)
    sim_mat = np.zeros((shot_num1,shot_num2))
    for i in range(shot_num1):
        for j in range(shot_num2):
            sim_mat[i][j] = concept_IOU(vTags[shot_seq1[i]],vTags[shot_seq2[j]])
    return sim_mat

def evaluation(pred_s1_lists, query_summary, Tags, test_vids, concepts, query_split, test_mode):
    # 首先根据两组concept_logits选出一组候选集，然后从候选里根据summary_logits做最终预测
    p_logits = pred_s1_lists[0]  # (bc*seq_len) * 48
    for i in range(1, len(pred_s1_lists)):
        p_logits = np.vstack((p_logits, pred_s1_lists[i]))

    pos = 0
    PRE_values = []
    REC_values = []
    F1_values = []
    for i in range(len(test_vids)):
        vid, vlength = test_vids[i]
        summary = query_summary[str(vid)]
        hl_num = math.ceil(vlength * 0.02)  # stage 2, 最终取2%作为summary
        p_predictions = p_logits[pos : pos + vlength]
        pos += vlength
        if not test_mode:
            queries = query_split[str(vid)]['valid_queries']
        else:
            queries = query_split[str(vid)]['test_queries']
        for query in queries:
            shots_gt = summary[query]
            c1, c2 = query.split('_')
            c1_ind = concepts.index(c1)
            c2_ind = concepts.index(c2)

            # make summary
            pred_c1 = p_predictions[:, c1_ind]
            pred_c2 = p_predictions[:, c2_ind]
            scores = (pred_c1 + pred_c2) / 2
            scores_indexes = scores.reshape((-1, 1))
            scores_indexes = np.hstack((scores_indexes, np.array(range(len(scores))).reshape((-1,1))))
            shots_pred = scores_indexes[scores_indexes[:, 0].argsort()]
            shots_pred = shots_pred[-hl_num:, 1].astype(int)
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

def run_testing(data_train, data_test, queries, query_summary, Tags, concepts, concept_embedding, segment_dict, model_path, query_split, test_mode):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        features_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_FEATURE))
        positions_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.seq_len))
        segment_embs_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.segment_num, D_FEATURE))
        segment_poses_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.segment_num))
        scores_src_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len + hp.segment_num + hp.memory_num))
        pred_s1_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_C_OUTPUT))
        dropout_holder = tf.placeholder(tf.float32, shape=())
        training_holder = tf.placeholder(tf.bool, shape=())

        # memory initialization
        if hp.memory_init == 'text' and hp.memory_num == CONCEPT_NUM and hp.memory_dimension == D_TXT_EMB:
            txt_embs = []
            for c in concepts:
                txt_embs.append(concept_embedding[c]['txt'])
            txt_embs = np.array(txt_embs).reshape([1, CONCEPT_NUM, D_TXT_EMB])
            memory_init = tf.constant_initializer(txt_embs)
        elif hp.memory_init == 'random':
            memory_init = tf.truncated_normal_initializer(stddev=0.01)
        memory_nodes_seq = tf.get_variable(name='memory_nodes',
                                           shape=(1, hp.memory_num, hp.memory_dimension),
                                           initializer=memory_init,
                                           dtype=tf.float32,
                                           trainable=True)
        memory_nodes = tf.tile(memory_nodes_seq, [hp.bc, 1, 1])

        # training operations
        lr = noam_scheme(hp.lr_noam, global_step, hp.warmup)
        opt_train = tf.train.AdamOptimizer(lr)

        # graph building
        tower_grads_train = []
        pred_s1_logits_list = []
        loss_list = []
        loss_ob_list = []
        for gpu_index in range(hp.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                features = features_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                positions = positions_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                segment_embs = segment_embs_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                segment_poses = segment_poses_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                scores_src = scores_src_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                pred_s1_labels = pred_s1_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]

                # 整合concept与summary的预测，形成最终预测
                pred_s1_logits, shots_output, memory_output = transformer(segment_embs, features, memory_nodes,
                                             segment_poses, positions, scores_src, dropout_holder, training_holder, hp, D_C_OUTPUT)
                pred_s1_logits_list.append(pred_s1_logits)

                  # 训练时每个序列只针对一个query预测summary
                loss, loss_ob = tower_loss_diverse(pred_s1_logits, pred_s1_labels, shots_output, memory_output, hp)
                varlist = tf.trainable_variables()  # 全部训练
                grads_train = opt_train.compute_gradients(loss, varlist)
                thresh = GRAD_THRESHOLD  # 梯度截断 防止爆炸
                grads_train_cap = [(tf.clip_by_value(grad, -thresh, thresh), var) for grad, var in grads_train]
                tower_grads_train.append(grads_train_cap)
                loss_list.append(loss)
                loss_ob_list += loss_ob
        grads_t = average_gradients(tower_grads_train)
        train_op = opt_train.apply_gradients(grads_t, global_step=global_step)
        if test_mode == 1:
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
        train_scheme = train_scheme_build(data_train, concepts, query_summary, hp)
        test_scheme, test_vids = test_scheme_build(data_test, hp.seq_len)
        epoch_step = math.ceil(len(train_scheme) / (hp.gpu_num * hp.bc))
        max_test_step = math.ceil(len(test_scheme) / (hp.gpu_num * hp.bc))

        # begin training
        ob_loss = []
        ob_sub_loss = []
        timepoint = time.time()
        for step in range(hp.maxstep):
            features_b, positions_b, segment_embs_b, segment_poses_b, scores_b, s1_labels_b = \
                get_batch_train(data_train, segment_dict, train_scheme, step, hp)
            scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, hp.segment_num + hp.memory_num))))  # encoder中开放所有concept节点
            observe = sess.run([train_op] +
                               loss_list +
                               pred_s1_logits_list +
                               loss_ob_list,
                               feed_dict={features_holder: features_b,
                                          positions_holder: positions_b,
                                          scores_src_holder: scores_src_b,
                                          pred_s1_labels_holder: s1_labels_b,
                                          segment_embs_holder: segment_embs_b,
                                          segment_poses_holder: segment_poses_b,
                                          dropout_holder: hp.dropout,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1 + hp.gpu_num])
            sub_loss_batch = observe[-3:]
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均
            ob_sub_loss.append(sub_loss_batch)

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == hp.maxstep:
                if step == 0 and test_mode == 0:
                    continue
                train_scheme = train_scheme_build(data_train, concepts, query_summary, hp)  # shuffle train scheme
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
                logging.info('S1_Loss: %.4f Shots_Diverse_Loss: %.4f Memory_Diverse_Loss: %.4f'%
                             (sub_loss_array[0], sub_loss_array[1], sub_loss_array[2]))
                if step < hp.protection or not int(epoch) % hp.eval_epoch == 0:
                    continue  # 增大测试间隔
                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                pred_s1_lists = []
                for test_step in range(max_test_step):
                    features_b, positions_b, segment_embs_b, segment_poses_b, scores_b = \
                        get_batch_test(data_test, segment_dict, test_scheme, test_step, hp)
                    scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, hp.segment_num + hp.memory_num))))  # encoder中开放所有concept节点
                    temp_list = sess.run(pred_s1_logits_list,feed_dict={features_holder: features_b,
                                                                        positions_holder: positions_b,
                                                                        scores_src_holder: scores_src_b,
                                                                        segment_embs_holder: segment_embs_b,
                                                                        segment_poses_holder: segment_poses_b,
                                                                        dropout_holder: hp.dropout,
                                                                        training_holder: False})
                    for preds in temp_list:
                        pred_s1_lists.append(preds.reshape((-1, D_C_OUTPUT)))

                # p, r, f = evaluation(pred_scores, queries, query_summary, Tags, test_vids, concepts)
                p, r, f = evaluation(pred_s1_lists, query_summary, Tags, test_vids, concepts, query_split, test_mode)
                logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f))
                return f
    return 0

def main(self):
    Tags = load_Tags(TAGS_PATH)
    data = load_feature_4fold(FEATURE_BASE, S1_LABEL_PATH, Tags)
    queries, query_summary = load_query_summary(QUERY_SUM_BASE)
    concepts, concept_embedding = load_concept(CONCEPT_DICT_PATH, CONCEPT_TXT_EMB_PATH, CONCEPT_IMG_EMB_DIR)
    segment_dict = segment_embedding_build(data, hp)
    with open(QUERY_SPLIT_PATH, 'r') as file:
        query_split = json.load(file)

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
                                 segment_dict, model_path, query_split, 1)
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