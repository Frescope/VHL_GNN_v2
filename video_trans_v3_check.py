# 基于trans2的两阶段模型改进，模型输出三种预测：concept，summary，prediction
# concept使用Tags作为标签训练，用于确定每个shot都与哪些concept相关，输出N*48的矩阵
# summary使用generic summary标签训练，用于找出每个shot是否是属于summary序列的，输出长为N的向量
# prediction使用s1，s2标签训练，用于做出最终预测，输出N*48的矩阵，对这一矩阵使用s1标签训练；在每次训练时计算对特定query的最终预测后用s2标签训练
# 构建候选集：使用soft（加权求和）的方法构建候选集，将concept的结果与summary的结果做归一化，用取max的方法合并，作为每个shot的soft-候选集得分
# 最终预测过程：在prediction的结果中取出与query相关的两组预测，求和后加上soft-候选集得分，作为最终预测
# 沿用两阶段模型的序列构建方式，对于两个concept(s1)与query(s2)本身分别提取正例与负例

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
from transformer_v3 import transformer
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

    # Encoder结构参数
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=6,type=int)

    # 序列参数，长度与正样本比例
    parser.add_argument('--seq_len',default=25,type=int)  # 视频序列的长度
    parser.add_argument('--qs_pr', default=0.1, type=float)  # query-summary positive ratio
    parser.add_argument('--concept_pr', default=0.5, type=float)

    # loss参数，不同loss所占比例
    parser.add_argument('--loss_concept_ratio', default=0.10, type=float)  # loss中来自concept_loss的比例
    parser.add_argument('--loss_summary_ratio', default=0.10, type=float)  # loss中来自summary_loss的比例
    parser.add_argument('--loss_pred_s1_ratio', default=0.50, type=float)  # loss中来自prediction_s1的比例
    parser.add_argument('--loss_pred_s2_ratio', default=0.30, type=float)  # loss中来自prediction_s2的比例
    parser.add_argument('--loss_diverse_ratio', default=0.05, type=float)  # loss中来自diverse_loss的比例

    # 预测参数，不同分支在总预测中的比例
    parser.add_argument('--pred_candidate_ratio', default=0.25, type=float)  # prediction中来自候选集得分的比例
    parser.add_argument('--pred_prediction_ratio', default=0.75, type=float)  # prediction中来自prediction分支的比例

    # 全局嵌入节点参数
    parser.add_argument('--global_ratio', default=0.2, type=float)  # 全局嵌入的抽样比例
    parser.add_argument('--global_mode', default='min', type=str)  # 全局嵌入的类型

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
# D_FEATURE = 2048  # for resnet
D_FEATURE = 1024  # for I3D
D_TXT_EMB = 300
D_IMG_EMB = 2048
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
    S1_LABEL_PATH = r'/public/data1/users/hulinkang/utc/videotrans_label_s1.json'
    S2_LABEL_PATH = r'/public/data1/users/hulinkang/utc/videotrans_label_s2.json'
    SUMMARY_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/summary_label.json'
    QUERY_SUM_BASE = r'/public/data1/users/hulinkang/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/public/data1/users/hulinkang/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'/public/data1/users/hulinkang/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/public/data1/users/hulinkang/utc/concept_embeddding/'
    MODEL_SAVE_BASE = r'/public/data1/users/hulinkang/model_HL_utc_query/'
    CKPT_MODEL_PATH = r'/public/data1/users/hulinkang/model_HL_utc_query/video_trans/'
else:
    # path for USTC servers
    FEATURE_BASE = r'/data/linkang/VHL_GNN/utc/features/'
    TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
    S1_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_s1.json'
    S2_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_s2.json'
    SUMMARY_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/summary_label.json'
    QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'/data/linkang/VHL_GNN/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/data/linkang/VHL_GNN/utc/concept_embeddding/'
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

def load_feature_4fold(feature_base, s1_label_path, s2_label_path, summary_label_path, Tags):
    # 注意label_s1对应的concept是按照字典序排列的，label_s2对应的query也是按照短语的字典序排列的
    # 加载特征 & 用于一阶段concept预测任务的标签 & 用于二阶段summary预测任务的标签
    with open(s1_label_path, 'r') as file:
        s1_labels = json.load(file)
    with open(s2_label_path, 'r') as file:
        s2_labels = json.load(file)
    with open(summary_label_path, 'r') as file:
        summary_labels = json.load(file)

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

        # label for s2
        s2_label = np.array(s2_labels[str(vid)])[:, :vlength].T
        data[str(vid)]['s2_label'] = s2_label

        # label for generic summary
        summary_label = np.array(summary_labels[str(vid)])
        data[str(vid)]['summary_label'] = summary_label

        # label for concept(Tags)
        concept_label = Tags[vid - 1]
        data[str(vid)]['concept_label'] = concept_label

        logging.info('Vid: '+str(vid)+
                     ' Feature: '+str(feature.shape)+
                     ' S1 Label: '+str(s1_label.shape)+
                     ' S2 Label: '+str(s2_label.shape)+
                     ' Summary Label: '+str(summary_label.shape)+
                     ' Concept Label: ' + str(concept_label.shape)
                     )
    return data

def get_global_embeddings(data, ratio, mode):
    # 根据一定的比例从原视频的特征中随机采样一些，按照给定的模式拼接
    vlength = len(data['concept_label'])
    sample_num = min(math.ceil(vlength * ratio), vlength)
    if sample_num >= vlength:
        embs_matrix = data['feature']
    else:
        samples = set()
        while len(samples) < sample_num:
            samples.add(np.random.randint(0, vlength))
        embs_matrix = data['feature'][list(samples)]  # n*D
    if mode == 'mean':
        embs = np.mean(embs_matrix, axis=0)
    elif mode == 'max':
        embs = np.max(embs_matrix, axis=0)
    elif mode == 'min':
        embs = np.min(embs_matrix, axis=0)
    else:
        embs = np.mean(embs_matrix, axis=0)
    embs = embs.reshape((1, D_FEATURE))
    return embs

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

def get_batch_train(data_train, train_scheme, queries, concepts, step, hp):
    # 从train_scheme中获取gpu_num*bc个序列，每个长度seq_len，并返回每个clip的全局位置
    batch_num = hp.gpu_num * hp.bc
    features = []
    global_embs = []
    concept_labels = []
    summary_labels = []
    s1_labels = []
    s2_labels = []
    positions = []
    qc_indexes = []  # 用于在训练时确定每个序列对应的query与concept的分别的索引，用于合并两种logits
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(train_scheme)
        vid, query, clip_list = train_scheme[pos]
        features.append(data_train[vid]['feature'][clip_list])
        global_embs.append(get_global_embeddings(data_train[vid], hp.global_ratio, hp.global_mode))
        s1_labels.append(data_train[vid]['s1_label'][clip_list])
        s2_labels.append(data_train[vid]['s2_label'][clip_list])
        concept_labels.append(data_train[vid]['concept_label'][clip_list])
        summary_labels.append(data_train[vid]['summary_label'][clip_list])
        positions.append(clip_list)
        q_ind = queries[vid].index(query)
        c1, c2 = query.split('_')
        c1_ind = concepts.index(c1)
        c2_ind = concepts.index(c2)
        qc_indexes.append([q_ind, c1_ind, c2_ind])
    features = np.array(features)
    global_embs = np.array(global_embs)
    s1_labels = np.array(s1_labels)
    s2_labels = np.array(s2_labels)
    concept_labels = np.array(concept_labels)
    summary_labels = np.array(summary_labels)
    positions = np.array(positions)
    qc_indexes = np.array(qc_indexes)
    scores = np.ones((batch_num, hp.seq_len + 1))  # 多一个全局嵌入
    return global_embs, features, positions, scores, concept_labels, summary_labels, s1_labels, s2_labels, qc_indexes

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
        vid, seq_start, seq_end = test_scheme[pos]
        vlength = len(data_test[str(vid)]['s1_label'])
        padding_len = hp.seq_len - (seq_end - seq_start)
        global_embs.append(get_global_embeddings(data_test[vid], 1.0, hp.global_mode))
        feature = data_test[str(vid)]['feature'][seq_start:seq_end]
        position = np.array(list(range(seq_start, seq_end)))
        label = data_test[str(vid)]['s1_label'][seq_start:seq_end]
        score = np.ones(len(label) + 1)
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
    features = np.array(features)
    global_embs = np.array(global_embs)
    positions = np.array(positions)
    scores = np.array(scores)
    return global_embs, features, positions, scores

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

def tower_loss_v3(concept_logits, concept_labels,
                  summary_logits, summary_labels,
                  pred_s1_logits, pred_s1_labels,
                  pred_s2_logits, pred_s2_labels,
                  reconst_vecs, features, hp):
    # concept_logits & concept_labels: bc*seq_len*48
    # summary_logits & summary_labels: bc*seq_len
    # pred_s1_logits & pred_s1_labels: bc*seq_len*48
    # pred_s2_logits & pred_s2_labels: bc*seq_len
    # 对concept_loss，计算各个shot对所有concept的交叉熵
    # 对summary_loss，计算各个shot对generic_summary的交叉熵
    # 对s1_loss，计算各个shot在s1中所有concept相关预测上的NCE-Loss的均值
    # 对s2_loss，计算最终prediction在其取样时对应的query上的NCE-Loss（为了防止负例比例过高）
    # 对diversity_loss, 选出prediction较高的shot，增大它们的输出特征之间的差距
    # 合并上述loss

    # for concept
    concept_logits = tf.clip_by_value(concept_logits, 1e-6, 0.999999)
    concept_labels_bin = tf.cast(tf.cast(concept_labels, dtype=tf.bool), dtype=tf.float32)
    concept_loss = - concept_labels_bin * tf.log(concept_logits) - (1 - concept_labels_bin) * tf.log(1 - concept_logits)
    concept_loss = tf.reduce_mean(concept_loss)

    # for summary
    summary_logits = tf.clip_by_value(summary_logits, 1e-6, 0.999999)
    summary_loss = - summary_labels * tf.log(summary_logits) - (1 - summary_labels) * tf.log(1 - summary_logits)
    summary_loss = tf.reduce_mean(summary_loss)

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

    # for s2，实际上的GT标签，每次只计算一个query
    s2_labels_bin = tf.cast(tf.cast(pred_s2_labels, dtype=tf.bool), dtype=tf.float32)  # 转化为0-1形式，浮点数
    s2_nce_pos = tf.reduce_sum(tf.exp(s2_labels_bin * pred_s2_logits), axis=1)  # 分子
    s2_nce_pos -= tf.reduce_sum((1 - s2_labels_bin), axis=1)  # 减去负例（为零）取e后的值（为1）
    s2_nce_all = tf.reduce_sum(tf.exp(pred_s2_logits), axis=1)  # 分母
    s2_nce_loss = -tf.log((s2_nce_pos / s2_nce_all) + 1e-5)
    s2_loss = tf.reduce_mean(s2_nce_loss)

    # for diversity
    if hp.loss_diverse_ratio > 0.01:
        top_50 = tf.nn.top_k(pred_s2_logits, int(hp.seq_len / 2))
        top_indices = top_50.indices  # 没行（序列）前50%的索引
        KeyVecs = []  # 得分较高的shot对应的重建向量
        for i in range(top_indices.get_shape().as_list()[0]):
            KeyVecs.append(tf.gather(reconst_vecs[i:i+1], top_indices[i], axis=1))
        KeyVecs = tf.concat(KeyVecs, axis=0)
        KeyVecs_T = tf.transpose(KeyVecs, perm=(0, 2, 1))  # bc*D*seqlen
        Products = tf.matmul(KeyVecs, KeyVecs_T)  # 点积，bc*seqlen*seqlen

        Magnitude = tf.sqrt(tf.reduce_sum(tf.square(KeyVecs), axis=2, keep_dims=True))  # 求模，bc*seqlen*1
        Magnitude_T = tf.transpose(Magnitude, perm=(0, 2, 1))  # bc*1*seqlen
        Mag_product = tf.matmul(Magnitude, Magnitude_T)  # bc*seqlen*seqlen

        diverse_loss = tf.reduce_sum(Products / (Mag_product + 1e-8), [1,2])
        diverse_loss = tf.reduce_mean(diverse_loss / (hp.seq_len - 1) / hp.seq_len)
    else:
        diverse_loss = tf.convert_to_tensor(np.zeros(1), dtype=tf.float32)

    # total loss
    loss = concept_loss * hp.loss_concept_ratio + \
           summary_loss * hp.loss_summary_ratio + \
           s1_loss * hp.loss_pred_s1_ratio + \
           s2_loss * hp.loss_pred_s2_ratio + \
           diverse_loss * hp.loss_diverse_ratio
    return loss, [concept_loss, summary_loss, s1_loss, s2_loss, diverse_loss]

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

def evaluation_2stages(concept_lists, summary_lists, pred_s1_lists, query_summary, Tags, test_vids, concepts, queries):
    # 首先根据两组concept_logits选出一组候选集，然后从候选里根据summary_logits做最终预测

    def MM_norm(preds):
        # 1D min-max normalization
        return (preds - preds.min()) / (preds.max() - preds.min())

    c_logits = concept_lists[0]  # (bc*seq_len) * 48
    s_logits = summary_lists[0]  # (bc*seq_len)
    p_logits = pred_s1_lists[0]  # (bc*seq_len) * 48
    for i in range(1, len(concept_lists)):
        c_logits = np.vstack((c_logits, concept_lists[i]))
        s_logits = np.hstack((s_logits, summary_lists[i]))
        p_logits = np.vstack((p_logits, pred_s1_lists[i]))

    pos = 0
    PRE_values = []
    REC_values = []
    F1_values = []
    for i in range(len(test_vids)):
        vid, vlength = test_vids[i]
        summary = query_summary[str(vid)]
        hl_num = math.ceil(vlength * 0.02)  # stage 2, 最终取2%作为summary
        c_predictions = c_logits[pos : pos + vlength]
        s_predictions = s_logits[pos : pos + vlength]
        p_predictions = p_logits[pos : pos + vlength]
        pos += vlength
        for query in summary:
            shots_gt = summary[query]
            q_ind = queries[vid].index(query)
            c1, c2 = query.split('_')
            c1_ind = concepts.index(c1)
            c2_ind = concepts.index(c2)

            # compute soft-candidate scores
            concept_c1 = MM_norm(c_predictions[:, c1_ind]).reshape((-1, 1))
            concept_c2 = MM_norm(c_predictions[:, c2_ind]).reshape((-1, 1))
            sum_generic = MM_norm(s_predictions).reshape((-1, 1))
            candidate = np.hstack((concept_c1, concept_c2, sum_generic))
            candidate = np.max(candidate, axis=1)

            # make summary
            pred_c1 = p_predictions[:, c1_ind]
            pred_c2 = p_predictions[:, c2_ind]
            scores = (pred_c1 + pred_c2) / 2 * hp.pred_prediction_ratio + \
                    candidate * hp.pred_candidate_ratio
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
def make_summary(concept_logits, summary_logits, pred_s1_logits, s2_labels, qc_indexes, hp):
    # 从concept_logits找出与query相关的两行，分别与summary_logits取max，得到soft-候选集得分，再与pred_s1中相关的预测相加求平均
    # input:
    # concept_logits: bc*seq_len*48
    # summary_logits: bc*seq_len
    # pred_s1_logits: bc*seq_len*48
    # s2_labels: bc*seq_len*45
    # output:
    # pred_s2_logits & labels: bc*seq_len

    def MM_norm(logits):
        # 1D min-max normalization
        logits_min = tf.reduce_min(logits)
        logits_max = tf.reduce_max(logits)
        return (logits - logits_min) / (logits_max - logits_min)

    pred_s2_logits = []  # 每个序列都只对应一个特定的query
    pred_s2_labels = []
    for i in range(hp.bc):
        s_ind = qc_indexes[i][0]
        c1_ind = qc_indexes[i][1]
        c2_ind = qc_indexes[i][2]
        summary = MM_norm(summary_logits[i])
        concept_c1 = MM_norm(concept_logits[i, :, c1_ind])
        concept_c2 = MM_norm(concept_logits[i, :, c2_ind])
        candidate_c1 = tf.where(tf.greater(summary, concept_c1), summary, concept_c1)  # 满足concept相关性或summary其中任一条件即可
        candidate_c2 = tf.where(tf.greater(summary, concept_c2), summary, concept_c2)
        candidate = tf.where(tf.greater(candidate_c1, candidate_c2), candidate_c1, candidate_c2)

        ps1_c1 = pred_s1_logits[i, :, c1_ind]
        ps1_c2 = pred_s1_logits[i, :, c2_ind]

        ps2 = (ps1_c1 + ps1_c2) / 2 * hp.pred_prediction_ratio + \
              candidate * hp.pred_candidate_ratio
        pred_s2 = tf.expand_dims(ps2, 0)
        s2_label = s2_labels[i : i + 1, :, s_ind]
        pred_s2_logits.append(pred_s2)
        pred_s2_labels.append(s2_label)

    pred_s2_logits = tf.concat(pred_s2_logits, axis=0)
    pred_s2_labels = tf.concat(pred_s2_labels, axis=0)
    return pred_s2_logits, pred_s2_labels

def run_testing(data_train, data_test, queries, query_summary, Tags, concepts, concept_embeeding, model_path):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        global_embs_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, 1, D_FEATURE))
        features_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_FEATURE))
        positions_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.seq_len))
        scores_src_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len + CONCEPT_NUM + 1))
        concept_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_C_OUTPUT))
        summary_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len))
        pred_s1_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_C_OUTPUT))
        s2_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_S_OUTPUT))
        qc_indexes_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, 3))  # 训练时给每个序列标记一个取样时的query与concept的索引
        txt_emb_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, CONCEPT_NUM, D_TXT_EMB))
        img_emb_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, CONCEPT_NUM, D_IMG_EMB))
        dropout_holder = tf.placeholder(tf.float32, shape=())
        training_holder = tf.placeholder(tf.bool, shape=())

        # training operations
        lr = noam_scheme(hp.lr_noam, global_step, hp.warmup)
        opt_train = tf.train.AdamOptimizer(lr)

        # graph building
        tower_grads_train = []
        concept_logits_list = []
        summary_logits_list = []
        pred_s1_logits_list = []
        pred_s2_logits_list = []
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
                pred_s1_labels = pred_s1_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                s2_labels = s2_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                qc_indexes = qc_indexes_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                txt_emb = txt_emb_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                img_emb = img_emb_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]

                # 整合concept与summary的预测，形成最终预测
                concept_logits, summary_logits, pred_s1_logits, reconst_vecs = \
                    transformer(global_embs, features, img_emb, positions, scores_src, dropout_holder, training_holder, hp, D_C_OUTPUT)
                pred_s2_logits, pred_s2_labels = make_summary(concept_logits, summary_logits, pred_s1_logits, s2_labels,
                                                              qc_indexes, hp)
                concept_logits_list.append(concept_logits)
                summary_logits_list.append(summary_logits)
                pred_s1_logits_list.append(pred_s1_logits)
                pred_s2_logits_list.append(pred_s2_logits)

                  # 训练时每个序列只针对一个query预测summary
                loss, loss_ob = tower_loss_v3(concept_logits, concept_labels,
                                              summary_logits, summary_labels,
                                              pred_s1_logits, pred_s1_labels,
                                              pred_s2_logits, pred_s2_labels,
                                              reconst_vecs, features, hp)
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
        train_scheme = train_scheme_build(data_train, concepts, query_summary, hp)
        test_scheme, test_vids = test_scheme_build(data_test, hp.seq_len)
        epoch_step = math.ceil(len(train_scheme) / (hp.gpu_num * hp.bc))
        max_test_step = math.ceil(len(test_scheme) / (hp.gpu_num * hp.bc))

        # concept embedding processing
        txt_emb_b = []
        img_emb_b = []
        for c in concepts:
            txt_emb_b.append(concept_embeeding[c]['txt'])
            img_emb_b.append(concept_embeeding[c]['img'])
        txt_emb_b = np.array(txt_emb_b).reshape([1, CONCEPT_NUM, D_TXT_EMB])
        img_emb_b = np.array(img_emb_b).reshape([1, CONCEPT_NUM, D_IMG_EMB])
        txt_emb_b = np.tile(txt_emb_b, [hp.gpu_num * hp.bc, 1, 1])  # (bc*gpu_num)*48*d_txt
        img_emb_b = np.tile(img_emb_b, [hp.gpu_num * hp.bc, 1, 1])

        # begin training
        ob_loss = []
        ob_sub_loss = []
        timepoint = time.time()
        for step in range(hp.maxstep):
            global_embs_b, features_b, positions_b, scores_b, concept_labels_b, summary_labels_b, s1_labels_b, s2_labels_b, query_indexes_b = \
                get_batch_train(data_train, train_scheme, queries, concepts, step, hp)
            scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, CONCEPT_NUM))))  # encoder中开放所有concept节点
            observe = sess.run([train_op] +
                               loss_list +
                               concept_logits_list +
                               summary_logits_list +
                               pred_s1_logits_list +
                               pred_s2_logits_list +
                               loss_ob_list,
                               feed_dict={features_holder: features_b,
                                          global_embs_holder: global_embs_b,
                                          positions_holder: positions_b,
                                          scores_src_holder: scores_src_b,
                                          concept_labels_holder: concept_labels_b,
                                          summary_labels_holder: summary_labels_b,
                                          pred_s1_labels_holder: s1_labels_b,
                                          s2_labels_holder: s2_labels_b,
                                          qc_indexes_holder: query_indexes_b,
                                          txt_emb_holder: txt_emb_b,
                                          img_emb_holder: img_emb_b,
                                          dropout_holder: hp.dropout,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1 + hp.gpu_num])
            sub_loss_batch = observe[-5:]
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均
            ob_sub_loss.append(sub_loss_batch)

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == hp.maxstep:
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
                logging.info('Concept_Loss: %.4f Summary_Loss: %.4f S1_Loss: %.4f S2_Loss: %.4f Diverse_Loss: %.4f' %
                             (sub_loss_array[0],sub_loss_array[1],sub_loss_array[2],sub_loss_array[3],sub_loss_array[4]))
                if not int(epoch) % hp.eval_epoch == 0:
                    continue  # 增大测试间隔
                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                concept_lists = []
                summary_lists = []
                pred_s1_lists = []
                for test_step in range(max_test_step):
                    global_embs_b, features_b, positions_b, scores_b = \
                        get_batch_test(data_test, test_scheme, test_step, hp)
                    scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, CONCEPT_NUM))))  # encoder中开放所有concept节点
                    temp_list = sess.run(concept_logits_list + summary_logits_list + pred_s1_logits_list,
                                                             feed_dict={features_holder: features_b,
                                                                        global_embs_holder: global_embs_b,
                                                                        positions_holder: positions_b,
                                                                        scores_src_holder: scores_src_b,
                                                                        txt_emb_holder: txt_emb_b,
                                                                        img_emb_holder: img_emb_b,
                                                                        dropout_holder: hp.dropout,
                                                                        training_holder: False})
                    for preds in temp_list[ : hp.gpu_num]:
                        concept_lists.append(preds.reshape((-1, D_C_OUTPUT)))
                    for preds in temp_list[hp.gpu_num : 2 * hp.gpu_num]:
                        summary_lists.append(preds.reshape((-1, )))
                    for preds in temp_list[2 * hp.gpu_num : ]:
                        pred_s1_lists.append(preds.reshape((-1, D_C_OUTPUT)))

                # p, r, f = evaluation(pred_scores, queries, query_summary, Tags, test_vids, concepts)
                p, r, f = evaluation_2stages(concept_lists, summary_lists, pred_s1_lists, query_summary, Tags, test_vids, concepts, queries)
                logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f))
                return f
    return 0


def main(self):
    Tags = load_Tags(TAGS_PATH)
    data = load_feature_4fold(FEATURE_BASE, S1_LABEL_PATH, S2_LABEL_PATH, SUMMARY_LABEL_PATH, Tags)
    queries, query_summary = load_query_summary(QUERY_SUM_BASE)
    concepts, concept_embedding = load_concept(CONCEPT_DICT_PATH, CONCEPT_TXT_EMB_PATH, CONCEPT_IMG_EMB_DIR)

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
        logging.info('Loss Summary Ratio: ' + str(hp.loss_summary_ratio))
        logging.info('Loss S1 Ratio: ' + str(hp.loss_pred_s1_ratio))
        logging.info('Loss S2 Ratio: ' + str(hp.loss_pred_s2_ratio))
        logging.info('Loss Diverse Ratio: ' + str(hp.loss_diverse_ratio))
        logging.info('Pred Candidate Ratio: ' + str(hp.pred_candidate_ratio))
        logging.info('Pred Prediction Ratio: ' + str(hp.pred_prediction_ratio))

        logging.info('Global Embedding Ratio: ' + str(hp.global_ratio))
        logging.info('Global Embedding Mode: ' + str(hp.global_mode))
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

