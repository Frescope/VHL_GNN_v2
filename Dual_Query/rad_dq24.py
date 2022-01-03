# 用于实验rad数据集

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
from transformer_uniset_dq24 import transformer
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Path:
    parser = argparse.ArgumentParser()
    # 显卡，服务器与存储
    parser.add_argument('--gpu', default='2',type=str)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--msd', default='rad_test', type=str)

    # 训练参数
    parser.add_argument('--bc',default=20,type=int)
    parser.add_argument('--dropout',default=0.1,type=float)
    parser.add_argument('--lr_noam', default=1e-4, type=float)
    parser.add_argument('--warmup', default=500, type=int)
    parser.add_argument('--maxstep', default=1000, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--observe', default=0, type=int)
    parser.add_argument('--eval_epoch', default=10, type=int)
    parser.add_argument('--start', default='00', type=str)
    parser.add_argument('--end', default='99', type=str)
    parser.add_argument('--protection', default=0, type=int)  # 不检查步数太小的模型
    parser.add_argument('--run_mode', default='train', type=str)  # train: 做训练，最后全部测试一次；test：只做测试
    parser.add_argument('--metrics', default='s_cor', type=str)  # p,r,f,k_cor,s_cor 选择一种主要的评价指标

    # Encoder结构参数
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=6,type=int)
    parser.add_argument('--num_blocks_local', default=3, type=int)  # local attention的层数
    parser.add_argument('--local_attention_pose', default='early', type=str)  # late & early，local attention的位置，前融合或后融合

    # 序列参数，长度与正样本比例
    parser.add_argument('--shot_num',default=10,type=int)  # 片段数量
    parser.add_argument('--shot_len',default=15, type=int)  # 片段帧数
    parser.add_argument('--pos_ratio', default=0.50, type=float)  # positive sample ratio

    # score参数
    parser.add_argument('--aux_pr', default=0.5, type=float)  # 用于dual_query的辅助得分比例

    # segment-embedding参数
    parser.add_argument('--segment_num', default=10, type=int)  # segment节点数量
    parser.add_argument('--segment_mode', default='min', type=str)  # segment-embedding的聚合方式

    # query-embedding参数
    parser.add_argument('--query_num', default=137, type=int)  # query节点数量，对应三个数据集中的query总数

    # memory参数
    parser.add_argument('--memory_num', default=20, type=int)  # memory节点数量
    parser.add_argument('--memory_dimension', default=1024, type=int)  # memory节点的维度

    # loss参数
    parser.add_argument('--loss_pred_ratio', default=0.80, type=float)  # pred损失比例
    parser.add_argument('--mem_div', default=0.10, type=float)  # memory_diversity损失比例
    parser.add_argument('--shots_div', default=0.10, type=float)  # shots_diversity损失比例

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
D_VISUAL = 512
D_QUERY = 512
MAX_VALUE = 0.01
GRAD_THRESHOLD = 10.0  # gradient threshold2

EVALUATE_MODEL = True
MIN_TRAIN_STEPS = 0
PRESTEPS = 0

if hp.server == 0:
    # path for JD server
    UNISET_BASE = r'/public/data1/users/hulinkang/Uniset/'
    MODEL_SAVE_BASE = r'/public/data1/users/hulinkang/model_HL_Unify/'
elif hp.server == 1:
    # path for USTC servers
    UNISET_BASE = r'/data/linkang/Uniset/'
    MODEL_SAVE_BASE = r'/data/linkang/model_HL_v4/'

logging.basicConfig(level=logging.INFO)

def load_query(uniset_base):
    with open(uniset_base + 'rad_clip_text.pkl', 'rb') as file:
        rad = pickle.load(file)
    query_embedding = {}
    for query in rad:
        text = rad[query]['text']
        embedding = rad[query]['feature']
        if text not in query_embedding:
            query_embedding[text] = embedding
    queries = list(query_embedding.keys())
    queries.sort()
    return queries, query_embedding, rad

def segment_embedding(feature):
    if hp.segment_num == 0:
        segments = np.zeros((0, D_VISUAL))
        poses = np.zeros((0, ))
        return segments, poses

    vlength = len(feature)
    seglength = math.floor(vlength / hp.segment_num)
    segments = []
    poses = []
    for i in range(hp.segment_num):
        start = seglength * i
        end = min(vlength, seglength * (i + 1))
        mid = math.ceil((start + end) / 2)
        poses.append(mid)
        embs = feature[start : end]
        if hp.segment_mode == 'mean':
            embs = np.mean(embs, axis=0)
        elif hp.segment_mode == 'max':
            embs = np.max(embs, axis=0)
        elif hp.segment_mode == 'min':
            embs = np.min(embs, axis=0)
        else:
            embs = np.mean(embs, axis=0)
        segments.append(embs)
    poses = np.array(poses)
    segments = np.array(segments)
    return segments, poses

def load_data(uniset_base, hp):
    # 加载三个数据集的特征和原始标签
    # load labels
    with open(uniset_base + 'rad_labels.json', 'r') as file:
        rad_labels = json.load(file)

    # load text embeddings
    queries, query_embedding, tvsum_dict = load_query(uniset_base)  # 需要额外使用类型-文本映射关系

    # load features & organize data
    data = {}
    name = 'rad'
    data[name] = {}
    for query in rad_labels:
        vid = rad_labels[query]['vid']
        data[name][vid] = {}
        # label
        label_line = np.array(rad_labels[query]['scores'])
        data[name][vid]['query_text'] = query
        data[name][vid]['frame_label'] = label_line.reshape((-1,))
        # feature
        feature_path = uniset_base + name + '_clip_visual_1fps/%s_CLIP_2fps.npy' % vid
        data[name][vid]['feature'] = np.load(feature_path)[:len(label_line)]
        # segment
        if hp.segment_num == 0:
            segments = np.zeros((0, D_VISUAL))
            poses = np.zeros((0))
        else:
            segments, poses = segment_embedding(data[name][vid]['feature'])
        data[name][vid]['segment_emb'] = segments
        data[name][vid]['segment_pos'] = poses
        # testing labels
        shot_scores = frame2shot(label_line.reshape((1, -1)), hp)
        # shot_binary = []
        # for score in shot_scores:
        #     hlnum = math.ceil(len(score) * 0.15)
        #     score_list = list(score)
        #     score_list.sort(reverse=True)
        #     threshold = score_list[hlnum]
        #     label_bin = np.zeros_like(score)
        #     for i in range(len(label_bin)):
        #         if score[i] > threshold and np.sum(label_bin) < hlnum:
        #             label_bin[i] = 1
        #     shot_binary.append(label_bin)
        data[name][vid]['shot_score'] = shot_scores
        # data[name][vid]['shot_binary'] = np.array(shot_binary)

        logging.info('Dataset: ' + str(name) +
                     ' Vid: ' + str(vid) +
                     ' Feature: ' + str(data[name][vid]['feature'].shape) +
                     ' Label: ' + str(data[name][vid]['frame_label'].shape) +
                     ' Segments: ' + str(segments.shape) +
                     ' Poses: ' + str(poses.shape) +
                     ' Shot Score: ' + str(data[name][vid]['shot_score'].shape) +
                     # ' Shot Bin: ' + str(data[name][vid]['shot_binary'].shape) +
                     ' Query: ' + str(data[name][vid]['query_text'])
                     )

    # split data
    K = 5
    data_split = {}
    rad_videos = list(data['rad'].keys())
    rad_videos.sort()
    for i in range(K):
        data_split[i] = {}
        # rad
        rad_num = math.ceil(len(rad_videos) / K)
        temp_list = rad_videos[i * rad_num: (i + 1) * rad_num]
        for vid in temp_list:
            data_split[i][vid] = data['rad'][vid]

    return data_split, rad_labels, queries, query_embedding

def train_scheme_build(data_train, hp):
    # 首先对每个视频计算其片段总数
    # 然后对每个序列挑选若干片段，维护一个列表，标记在每次scheme_build时一个片段是否被选中过，支持重复选择之前序列中用过的片段
    # 对选中的片段，找出其对应的帧、位置及标签，并生成padding标记

    train_scheme = []
    for vid in data_train:
        vlength = len(data_train[vid]['feature'])
        nums = math.ceil(vlength / hp.shot_len)  # 一个视频中的片段总数，取上整
        if nums < hp.shot_num:  # 一个视频凑不齐一个序列
            train_scheme.append((vid, list(range(nums))))
            continue
        seq_num = math.ceil(nums / hp.shot_num)  # 这一视频能组成的序列总数，取上整
        shot_list = list(range(nums))  # 打乱片段顺序
        random.shuffle(shot_list)

        pos = 0
        for i in range(seq_num - 1):  # 不需要padding的序列数量
            seq = []
            for _ in range(hp.shot_num):
                seq.append(shot_list[pos])  # 随机弹出shotnum个片段构成一个序列
                pos += 1
            seq.sort()
            train_scheme.append((vid, seq))  # 保存视频编号与片段编号

        seq = shot_list[pos : ]
        while len(seq) < hp.shot_num:  # 如果最后一个序列长度不足，重复挑选一些片段
            seq.append(shot_list.pop(0))
        seq.sort()
        train_scheme.append((vid, seq))

    return train_scheme

def get_batch_train(data_train, train_scheme, queries, step, hp):
    # 从train_scheme中获取gpu_num*bc个序列，每个长度shot_num，并返回每个frame的全局位置
    def shot2frame(shot_list, vlength):
        frame_list = []
        for shot in shot_list:
            st = shot*hp.shot_len
            ed = min((shot+1)*hp.shot_len, vlength)
            frame_list += list(range(st, ed))
        return frame_list

    batch_num = hp.gpu_num * hp.bc
    features = []
    labels = []
    positions = []
    segment_embs = []
    segment_poses = []
    indexes = []
    scores = []  # padding标记，对应到每一帧
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(train_scheme)
        vid, shot_list = train_scheme[pos]
        vlength = len(data_train[str(vid)]['feature'])
        frame_list = shot2frame(shot_list, vlength)
        padding_len = hp.shot_num * hp.shot_len - len(frame_list)  # 输入的一个序列长度是不定的，需要计算待补充的空白帧数

        feature = data_train[vid]['feature'][frame_list]
        label = data_train[vid]['frame_label'][frame_list]
        position = frame_list
        score = np.ones((len(frame_list)))
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_VISUAL))
            label_pad = np.zeros((padding_len))
            position_pad = np.array([vlength] * padding_len)
            score_pad = np.zeros(padding_len)
            feature = np.vstack((feature, feature_pad))
            label = np.hstack((label, label_pad))
            position = np.hstack((position, position_pad))
            score = np.hstack((score, score_pad))

        features.append(feature)
        labels.append(label)
        positions.append(position)
        scores.append(score)
        segment_embs.append(data_train[vid]['segment_emb'])
        segment_poses.append(data_train[vid]['segment_pos'])
        indexes.append(queries.index(data_train[vid]['query_text']))  # 对应的query文本顺序

    features = np.array(features)
    labels = np.array(labels)
    positions = np.array(positions)
    segment_embs = np.array(segment_embs)
    segment_poses = np.array(segment_poses)
    indexes = np.array(indexes)
    scores = np.array(scores)
    return features, positions, segment_embs, segment_poses, indexes, scores, labels

def test_scheme_build(data_test, hp):
    # 依次输入测试集中所有shot，不足shotnum的要补足，在getbatch中补足不够一个batch的部分
    # (vid, seq_start, seq_end)形式，编号对应到帧
    test_scheme = []
    test_vids = []
    frame_num = hp.shot_num * hp.shot_len  # 一个序列中所包含的帧数量
    for vid in data_test:
        vlength = len(data_test[str(vid)]['feature'])
        seq_num = math.ceil(vlength / frame_num)
        for i in range(seq_num):
            test_scheme.append((vid, i * frame_num, min(vlength, (i + 1) * frame_num)))
        test_vids.append((vid, math.ceil(vlength / hp.shot_len)))
    return test_scheme, test_vids

def get_batch_test(data_test, test_scheme, queries, step, hp):
    features = []
    positions = []
    segment_embs = []
    segment_poses = []
    indexes = []
    scores = []
    batch_num = hp.gpu_num * hp.bc
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(test_scheme)
        vid, seq_start, seq_end = test_scheme[pos]
        vlength = len(data_test[str(vid)]['feature'])
        padding_len = hp.shot_num * hp.shot_len - (seq_end - seq_start)  # 每一个序列中需要填充的空白帧数
        feature = data_test[str(vid)]['feature'][seq_start:seq_end]
        position = np.array(list(range(seq_start, seq_end)))
        indexes.append(queries.index(data_test[vid]['query_text']))
        score = np.ones(seq_end - seq_start)
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_VISUAL))
            position_pad = np.array([vlength] * padding_len)
            score_pad = np.zeros(padding_len)
            feature = np.vstack((feature, feature_pad))
            position = np.hstack((position, position_pad))
            score = np.hstack((score, score_pad))
        features.append(feature)
        positions.append(position)
        scores.append(score)
        segment_embs.append(data_test[vid]['segment_emb'])
        segment_poses.append(data_test[vid]['segment_pos'])
    features = np.array(features)
    positions = np.array(positions)
    segment_embs = np.array(segment_embs)
    segment_poses = np.array(segment_poses)
    indexes = np.array(indexes)
    scores = np.array(scores)
    return features, positions, segment_embs, segment_poses, indexes, scores

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

def tower_loss(pred_scores, pred_labels, shots_output, memory_output, hp):
    # pred_scores: dual_query_scores, bc*shotnum
    # pred_labels: bc*shot_num
    # shots_output: bc*shot_num*D （或是frame-level的输出，由聚合位置决定）
    # memory_output: bc*memory_num*D
    # 对pred_loss，计算每一帧在其视频对应的query上的NCE-Loss
    # 计算shots与memory的diversity-loss，pred_loss加权求和

    def cosine_similarity(Vecs):
        # 计算一组向量中每个向量与其他所有向量的相似度，不包含自身
        # Vecs: bc*N*D
        Vecs_T = tf.transpose(Vecs, perm=(0, 2, 1))  # bc*D*N
        Products = tf.matmul(Vecs, Vecs_T)  # 点积，bc*N*N

        Magnitude = tf.sqrt(tf.reduce_sum(tf.square(Vecs), axis=2, keep_dims=True))  # 求模，bc*N*1
        Magnitude_T = tf.transpose(Magnitude, perm=(0, 2, 1))  # bc*1*N
        Mag_product = tf.matmul(Magnitude, Magnitude_T)  # bc*N*N

        similarity = Products / (Mag_product + 1e-8) + 1  # bc*N*N，每个元素与其他任一元素的相似度，加一以避免负值
        similarity = tf.reduce_sum(similarity, axis=1, keepdims=True) - (1 / (1 + 1e-8) + 1)  # bc*1*N，每个元素与其他所有元素的相似度之和
        return similarity

    # def ranking_loss(score_list):
    #     # 输入一组序列，每个序列都从大到小排列，输出其
    # for pred，与分解到concept的query-summary label
    # labels_bin = tf.cast(tf.cast(pred_labels, dtype=tf.bool), dtype=tf.float32)  # 转化为0-1形式，浮点数
    #
    # pred_nce_pos = tf.reduce_sum(tf.exp(labels_bin * pred_scores), axis=1)  # 分子
    # pred_nce_pos -= tf.reduce_sum((1 - labels_bin), axis=1)  # 减去负例（为零）取e后的值（为1）
    # pred_nce_all = tf.reduce_sum(tf.exp(pred_scores), axis=1)  # 分母
    # pred_nce_loss = -tf.log((pred_nce_pos / pred_nce_all) + 1e-5)
    # pred_loss = tf.reduce_mean(pred_nce_loss)

    # ranking-loss ListMLE
    pred_scores_sort = []
    for i in range(hp.bc):
        indices = tf.nn.top_k(pred_labels[i], k=hp.shot_num).indices
        pred_scores_sort.append(tf.gather(pred_scores[i], indices))  # 排序后, bc*shotnum
    scores_sums = tf.reduce_sum(pred_scores, axis=1)
    pred_loss = 0
    loss_list = []
    for i in range(hp.bc):
        sum_temp = scores_sums[i]
        loss_temp = 1
        for j in range(hp.shot_num):
            item = pred_scores_sort[i][j] / (sum_temp + 1e-6)  # 当前要乘的一项
            loss_temp *= item
            sum_temp -= pred_scores_sort[i][j]  # 更新分母
        pred_loss += -tf.log(loss_temp)
        loss_list.append(loss_temp)
    pred_loss /= hp.bc

    # # ranking-loss ListNet
    # pred_scores_sort = []
    # pred_labels_sort = []
    # for i in range(hp.bc):
    #     indices = tf.nn.top_k(pred_labels[i], k=hp.shot_num).indices
    #     pred_scores_sort.append(tf.gather(pred_scores[i], indices))  # 排序后, bc*shotnum
    #     pred_labels_sort.append(tf.gather(pred_labels[i], indices))
    # scores_sums = tf.reduce_sum(pred_scores, axis=1)
    # labels_sums = tf.reduce_sum(pred_labels, axis=1)
    # pred_loss = 0
    # loss_list = []
    # for i in range(hp.bc):
    #     score_sum = scores_sums[i]
    #     score_product = 1
    #     for j in range(hp.shot_num):
    #         item = pred_scores_sort[i][j] / (score_sum + 1e-6)  # 当前要乘的一项
    #         score_product *= item
    #         score_sum -= pred_scores_sort[i][j]
    #     label_sum = labels_sums[i]
    #     label_product = 1
    #     for j in range(hp.shot_num):
    #         item = pred_labels_sort[i][j] / (label_sum + 1e-6)
    #         label_product *= item
    #         label_sum -= pred_labels_sort[i][j]
    #     loss_list.append((label_product, score_product))
    #     pred_loss += tf.exp(label_product) * tf.log((label_product + 1e-8) / score_product)
    # pred_loss /= hp.bc

    # shots多样性

    if hp.shots_div >= 0.05:
        length = pred_scores.get_shape().as_list()[1]
        soft_weights = tf.expand_dims(tf.nn.softmax(pred_scores, axis=1), axis=2)  # bc*shot_num*1，计算对于每个concept的重要性权重
        shots_similiarity = cosine_similarity(shots_output)  # bc*1*shot_num，计算每个片段特征对其他所有片段特征的相似度
        shots_similiarity = tf.matmul(shots_similiarity, soft_weights)  # bc*1*1，相似度加权求和
        shots_div_loss = tf.reduce_mean(shots_similiarity / length / (length - 1))  # 对所有concept的相似度损失求和，再对所有序列求平均
    else:
        shots_div_loss = tf.convert_to_tensor(np.zeros(1), dtype=tf.float32)

    # memory多样性
    if hp.mem_div >= 0.05:
        memory_similiarity = cosine_similarity(memory_output)  # bc*1*mem_len
        memory_div_loss = tf.reduce_mean(
            tf.reduce_sum(memory_similiarity, axis=2) / hp.memory_num / (hp.memory_num - 1))  # 对所有memory节点 的相似度损失求和，再对所有序列求平均
    else:
        memory_div_loss = tf.convert_to_tensor(np.zeros(1), dtype=tf.float32)

    loss = pred_loss * hp.loss_pred_ratio + \
           shots_div_loss * hp.shots_div + \
           memory_div_loss * hp.mem_div
    return loss, [pred_loss, shots_div_loss, memory_div_loss] + loss_list

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

def frame2shot(labels, hp):
    # 将帧标签变成片段标签，标签为N*vlength形式
    N = len(labels)
    vlength = len(labels[0])
    nums = math.floor(vlength / hp.shot_len)  # 视频中的片段数，取下整
    labels_shot = labels[:, : nums*hp.shot_len]
    labels_rem = labels[:, nums*hp.shot_len:]  # 切掉最后不足一个片段的几帧
    labels_shot = labels_shot.reshape((N, nums, hp.shot_len))
    labels_shot = np.mean(labels_shot, axis=2)  # N*nums
    if len(labels_rem[0]) > 0:
        labels_rem = np.mean(labels_rem, axis=1, keepdims=True)  # N*1（或N*0）
    else:
        labels_rem = np.zeros((N, 0))
    labels_new = np.concatenate([labels_shot, labels_rem],axis=1)  # N*(nums+1)
    return labels_new

def evaluation(pred_scores, test_videos, uniset_labels, data_test, hp):

    def compute(preds, vid):
        # 计算某个视频的f1值与ranking指标
        ranking_labels = data_test[vid]['shot_score']
        # binary_labels = data_test[vid]['shot_binary']

        # for f1
        # hlnum = math.ceil(len(preds) * 0.15)
        # preds_list = list(preds)
        # preds_list.sort(reverse=True)
        # threshold = preds_list[hlnum]
        # labels_pred = np.zeros_like(preds)
        # for i in range(len(labels_pred)):
        #     if preds[i] > threshold and np.sum(labels_pred) < hlnum:
        #         labels_pred[i] = 1
        p = r = f = 0  # 计算多标签的结果求均值
        # for label in binary_labels:
        #     p += precision_score(label, labels_pred)
        #     r += recall_score(label, labels_pred)
        #     f += f1_score(label, labels_pred)
        # p /= len(ranking_labels)
        # r /= len(ranking_labels)
        # f /= len(ranking_labels)
        # for ranking
        k_cor = 0
        s_cor = 0
        for label in ranking_labels:
            k_cor_tmp, _ = stats.kendalltau(preds, label)
            s_cor_tmp, _ = stats.spearmanr(preds, label)
            k_cor += k_cor_tmp
            s_cor += s_cor_tmp
        k_cor /= len(ranking_labels)
        s_cor /= len(ranking_labels)
        return p, r, f, k_cor, s_cor

    # 首先将模型输出裁剪为对每个视频中的帧得分预测
    preds_c = list(pred_scores[0])
    for i in range(1, len(pred_scores)):
        preds_c = preds_c + list(pred_scores[i])

    # 计算F1与相关性系数，并按照数据集分别归类
    pos = 0
    results = {
        'p': [], 'r': [], 'f': [], 'k_cor': [], 's_cor': []
    }
    for vid, num in test_videos:
        y_pred = np.array(preds_c[pos:pos + num])
        padlen = (hp.shot_num - num % hp.shot_num) % hp.shot_num  # 当vlength是seq_len的整数倍时，不需要padding
        pos += num + padlen  # 跳过padding部分
        p, r, f, k_cor, s_cor = compute(y_pred, vid)
        results['p'].append(p)
        results['r'].append(r)
        results['f'].append(f)
        results['k_cor'].append(k_cor)
        results['s_cor'].append(s_cor)

    # output
    for key in results:
        results[key] = sum(results[key]) / len(results[key])
    logging.info('TVSum F1(prf): %.3f %.3f %.3f, Rank(ks): %.3f %.3f' %
                 (results['p'], results['r'], results['f'], results['k_cor'], results['s_cor']))

    return results

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def model_clear(model_save_dir, max_value):
    # 清除之前所有F1较小的模型
    models = []
    for name in os.listdir(model_save_dir):
        if name.endswith('.meta'):
            models.append(name.split('.meta')[0])
    for model in models:
        value = model.split('-')[-1]
        if value.startswith('V') and float(value.split('V')[-1]) < max_value:
            file_path = os.path.join(model_save_dir, model) + '*'
            os.system('rm -rf %s' % file_path)
    return

def model_search(model_save_dir):
    # 找到要验证的模型名称
    model_to_restore = []
    for root,dirs,files in os.walk(model_save_dir):
        for file in files:
            if file.endswith('.meta'):
                model_name = file.split('.meta')[0]
                model_to_restore.append(os.path.join(root, model_name))
    model_to_restore = list(set(model_to_restore))
    return model_to_restore

def running(data_train, data_test, uniset_labels, queries, query_embedding, test_mode, model_save_dir, model_path):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    max_value = MAX_VALUE

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        features_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.shot_num*hp.shot_len, D_VISUAL))
        positions_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.shot_num*hp.shot_len))
        segment_embs_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.segment_num, D_VISUAL))
        segment_poses_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.segment_num))
        query_embs_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.query_num, D_QUERY))
        indexes_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, ))
        scores_src_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num,
                                                              hp.shot_num*hp.shot_len + hp.segment_num + hp.query_num + hp.memory_num
                                                              ))
        pred_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.shot_num*hp.shot_len))
        dropout_holder = tf.placeholder(tf.float32, shape=())
        training_holder = tf.placeholder(tf.bool, shape=())

        # query embeddings
        query_embs_b = []
        for c in queries:
            query_embs_b.append(query_embedding[c])
        query_embs_b = np.array(query_embs_b).reshape((1, hp.query_num, D_QUERY))
        query_embs_b = np.tile(query_embs_b, [hp.gpu_num * hp.bc, 1, 1])

        # memory initialization
        memory_init = tf.truncated_normal_initializer(stddev=0.01)
        memory_nodes_seq = tf.get_variable(name='memory_nodes',
                                           shape=(1, hp.memory_num, hp.memory_dimension),
                                           initializer=memory_init,
                                           dtype=tf.float32,
                                           trainable=True)
        memory_nodes = tf.tile(memory_nodes_seq, [hp.bc * hp.gpu_num, 1, 1])

        # training operations
        lr = noam_scheme(hp.lr_noam, global_step, hp.warmup)
        opt_train = tf.train.AdamOptimizer(lr)

        # graph building
        tower_grads_train = []
        pred_scores_list = []
        loss_list = []
        loss_ob_list = []
        for gpu_index in range(hp.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                features = features_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                positions = positions_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                segment_embs = segment_embs_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                segment_poses = segment_poses_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                query_embs = query_embs_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                indexes = indexes_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                scores_src = scores_src_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                pred_labels = pred_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]

                shot_output, memory_output, pred_scores = transformer(features, segment_embs,
                                                                                   query_embs, memory_nodes,
                                                                                   segment_poses, positions,
                                                                                   indexes,
                                                                                   scores_src, dropout_holder,
                                                                                   training_holder, hp)
                pred_labels = tf.reduce_mean(tf.reshape(pred_labels, shape=[hp.bc, hp.shot_num, hp.shot_len]), axis=2)  # bc*shotnum
                pred_scores_list.append(pred_scores)

                loss, loss_ob = tower_loss(pred_scores, pred_labels, shot_output, memory_output, hp)
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
        saver_overall = tf.train.Saver(max_to_keep=20)
        if test_mode:
            logging.info(' Ckpt Model Restoring: ' + model_path)
            saver_overall.restore(sess, model_path)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        train_scheme = train_scheme_build(data_train, hp)
        test_scheme, test_vids = test_scheme_build(data_test, hp)
        epoch_step = math.ceil(len(train_scheme) / (hp.gpu_num * hp.bc))
        max_test_step = math.ceil(len(test_scheme) / (hp.gpu_num * hp.bc))

        # begin training
        ob_loss = []
        ob_sub_loss = []
        timepoint = time.time()
        for step in range(hp.maxstep):
            features_b, positions_b, segment_embs_b, segment_poses_b, indexes_b, scores_b, s1_labels_b = \
                get_batch_train(data_train, train_scheme, queries, step, hp)
            scores_src_b = np.hstack(
                (scores_b, np.ones((hp.gpu_num * hp.bc, hp.segment_num + hp.query_num + hp.memory_num))))  # encoder中开放所有concept节点
            observe = sess.run([train_op] +
                               loss_list +
                               pred_scores_list +
                               loss_ob_list,
                               feed_dict={features_holder: features_b,
                                          positions_holder: positions_b,
                                          scores_src_holder: scores_src_b,
                                          pred_labels_holder: s1_labels_b,
                                          segment_embs_holder: segment_embs_b,
                                          segment_poses_holder: segment_poses_b,
                                          indexes_holder: indexes_b,
                                          query_embs_holder: query_embs_b,
                                          dropout_holder: hp.dropout,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1 + hp.gpu_num])
            sub_loss_batch = observe[3:6]
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均
            ob_sub_loss.append(sub_loss_batch)

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == hp.maxstep or test_mode == 1:
                if step == 0 and test_mode == 0:
                    continue
                train_scheme = train_scheme_build(data_train, hp)  # shuffle train scheme
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
                logging.info('S1_Loss: %.4f Shots_Diverse_Loss: %.4f Memory_Diverse_Loss: %.4f' %
                             (sub_loss_array[0], sub_loss_array[1], sub_loss_array[2]))
                if test_mode == 0 and (step < hp.protection or not int(epoch) % hp.eval_epoch == 0):
                    continue  # 增大测试间隔
                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                pred_s1_lists = []
                for test_step in range(max_test_step):
                    features_b, positions_b, segment_embs_b, segment_poses_b, indexes_b, scores_b = \
                        get_batch_test(data_test, test_scheme, queries, test_step, hp)
                    scores_src_b = np.hstack((scores_b, np.ones(
                        (hp.gpu_num * hp.bc, hp.segment_num + hp.query_num + hp.memory_num))))  # encoder中开放所有concept节点
                    temp_list = sess.run(pred_scores_list, feed_dict={features_holder: features_b,
                                                                      positions_holder: positions_b,
                                                                      scores_src_holder: scores_src_b,
                                                                      segment_embs_holder: segment_embs_b,
                                                                      segment_poses_holder: segment_poses_b,
                                                                      indexes_holder: indexes_b,
                                                                      query_embs_holder: query_embs_b,
                                                                      dropout_holder: hp.dropout,
                                                                      training_holder: False})
                    for preds in temp_list:
                        pred_s1_lists.append(preds.reshape((-1,)))

                results = evaluation(pred_s1_lists, test_vids, uniset_labels, data_test, hp)
                # logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f, K_cor: %.3f, S_cor: %.3f' %
                #              (results['p'], results['r'], results['f'], results['k_cor'], results['s_cor']))

                if test_mode == 1:
                    return results[hp.metrics]

                # save model
                if step > MIN_TRAIN_STEPS - PRESTEPS and results[hp.metrics] >= max_value:
                    max_value = results[hp.metrics]
                    model_clear(model_save_dir, max_value)
                    model_path = model_save_dir + 'S%d-E%d-L%.6f-V%.3f' % (step, epoch, np.mean(loss_array), max_value)
                    saver_overall.save(sess, model_path)
                    logging.info('Model Saved: ' + model_path + '\n')

            if step % 3000 == 0 and step > 0:
                model_path = model_save_dir + 'S%d-E%d' % (step + PRESTEPS, epoch)
                # saver_overall.save(sess, model_path)
                logging.info('Model Saved: ' + str(step + PRESTEPS))

            # saving final model
        model_path = model_save_dir + 'S%d' % (hp.maxstep + PRESTEPS)
        # saver_overall.save(sess, model_path)
        logging.info('Model Saved: ' + str(hp.maxstep + PRESTEPS))
    return 0

def main(self):

    def preprocess(data, k):
        # 划分数据集，输出信息
        train_split = [(k + 0) % 5, (k + 1) % 5, (k + 2) % 5]
        valid_split = [(k + 3) % 5]
        test_split = [(k + 4) % 5]
        data_train = {}
        data_valid = {}
        data_test = {}
        for i in train_split:
            data_train.update(data[i])
        for i in valid_split:
            data_valid.update(data[i])
        for i in test_split:
            data_test.update(data[i])

        logging.info('*' * 20 + 'Settings' + '*' * 20)
        logging.info('K-fold: ' + str(k))
        logging.info('Train: %s' % str(train_split))
        logging.info('Valid: %s  Test: %s' % (str(valid_split), str(test_split)))
        logging.info('Model Base: ' + MODEL_SAVE_BASE + hp.msd)
        logging.info('WarmUp: ' + str(hp.warmup))
        logging.info('Noam LR: ' + str(hp.lr_noam))
        logging.info('Num Heads: ' + str(hp.num_heads))
        logging.info('Num Blocks: ' + str(hp.num_blocks))
        logging.info('Main Metrics: ' + str(hp.metrics))
        logging.info('Batchsize: ' + str(hp.bc))
        logging.info('Max Steps: ' + str(hp.maxstep))
        logging.info('Dropout Rate: ' + str(hp.dropout))
        logging.info('Shot Number: ' + str(hp.shot_num))
        logging.info('Shot Length: ' + str(hp.shot_len))
        logging.info('Evaluation Epoch: ' + str(hp.eval_epoch))
        logging.info('Auxiliary Score Ratio: ' + str(hp.aux_pr))
        logging.info('Segment Nodes Number: ' + str(hp.segment_num))
        logging.info('Segment Aggregation Mode: ' + str(hp.segment_mode))
        logging.info('Memory Nodes Number: ' + str(hp.memory_num))
        logging.info('Memory Nodes Dimension: ' + str(hp.memory_dimension))
        logging.info('Loss Prediction Ratio: ' + str(hp.loss_pred_ratio))
        logging.info('Loss Memory Diversity Ratio: ' + str(hp.mem_div))
        logging.info('Loss Shots Diversity Ratio: ' + str(hp.shots_div))
        logging.info('*' * 50)

        return data_train, data_valid, data_test

    data, rad_labels, queries, query_embedding = load_data(UNISET_BASE, hp)

    # 全部训练一次
    logging.info('*' * 20 + 'Training:' + '*' * 20)
    kfold_start = int(int(hp.start) / 10)
    kfold_end = min(4, int(int(hp.end) / 10))
    repeat_start = int(hp.start) % 10
    repeat_end = min(hp.repeat, int(hp.end) % 10)
    for kfold in range(kfold_start, kfold_end + 1):
        if hp.run_mode == 'test':
            break
        data_train, data_valid, data_test = preprocess(data, kfold)
        # repeat
        for i in range(hp.repeat):
            if kfold == kfold_start and i < repeat_start:
                continue
            model_save_dir = MODEL_SAVE_BASE + hp.msd + '_%d_%d/' % (kfold, i)
            logging.info('*' * 10 + str(i) + ': ' + model_save_dir + '*' * 10)
            logging.info('*' * 60)
            running(data_train, data_valid, rad_labels, queries, query_embedding, 0, model_save_dir, '')
            logging.info('*' * 60)
            if kfold >= kfold_end and i >= repeat_end:
                break
        logging.info('^' * 60 + '\n')

    # 测试
    logging.info('*' * 20 + 'Testing:' + '*' * 20)
    model_scores = {}
    for kfold in range(5):
        data_train, data_valid, data_test = preprocess(data, kfold)
        scores = []
        for i in range(hp.repeat):
            model_save_dir = MODEL_SAVE_BASE + hp.msd + '_%d_%d/' % (kfold, i)
            models_to_restore = model_search(model_save_dir)
            for i in range(len(models_to_restore)):
                logging.info('-' * 20 + str(i) + ': ' + models_to_restore[i].split('/')[-1] + '-' * 20)
                model_path = models_to_restore[i]
                value = running(data_train, data_test, rad_labels, queries, query_embedding, 1, model_save_dir, model_path)
                scores.append(value)
        model_scores[kfold] = scores
    scores_all = 0
    for kfold in model_scores:
        scores = model_scores[kfold]
        logging.info('Vid: %s, Mean: %.3f, Scores: %s' %
                     (kfold, np.array(scores).mean(), str(scores)))
        scores_all += np.array(scores).mean()
    logging.info('Overall Results: %.3f' % (scores_all / len(model_scores)))

if __name__ == '__main__':
    tf.app.run()


