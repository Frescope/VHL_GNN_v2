# 用于personalized video summarization的naive版本
# 所有video中随机抽取一些帧，提取特征后做在数据集范围内的全局聚类，然后根据聚类结果给每帧打标签
# 使用Spearman & Kendall 相关系数作为评价指标
# 使用聚类标签和score_avg标签做训练，使用multiple GT标签做测试

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
from scipy import stats

from transformer_genericVS import transformer

class Path:
    parser = argparse.ArgumentParser()
    # 显卡，服务器与存储
    parser.add_argument('--gpu', default='0',type=str)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--msd', default='video_trans_test', type=str)

    # 训练参数
    parser.add_argument('--bc',default=10,type=int)
    parser.add_argument('--dropout',default='0.1',type=float)
    parser.add_argument('--lr_noam', default=1e-5, type=float)
    parser.add_argument('--warmup', default=5000, type=int)
    parser.add_argument('--maxstep', default=20000, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--observe', default=0, type=int)
    parser.add_argument('--eval_epoch', default=1, type=int)
    parser.add_argument('--start', default='00', type=str)
    parser.add_argument('--end', default='', type=str)
    parser.add_argument('--protection', default=00, type=int)  # 不检查步数太小的模型

    # Encoder结构参数
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=6,type=int)

    # 序列参数，长度与正样本比例
    parser.add_argument('--seq_len',default=25,type=int)  # 视频序列的长度
    parser.add_argument('--pos_ratio', default=0.2, type=float)  # positive sample ratio

    # segment-embedding参数
    parser.add_argument('--segment_num', default=10, type=int)  # segment节点数量
    parser.add_argument('--segment_mode', default='min', type=str)  # segment-embedding的聚合方式

    # memory参数
    parser.add_argument('--memory_num', default=60, type=int)  # memory节点数量
    parser.add_argument('--memory_dimension', default=1024, type=int)  # memory节点的维度
    parser.add_argument('--memory_init', default='random', type=str)  # random, text

    # loss参数
    parser.add_argument('--loss_cluster_ratio', default=0.50, type=float)  # cluster损失比例
    parser.add_argument('--loss_rank_ratio', default=0.50, type=float)  # rank损失比例
    parser.add_argument('--mem_div', default=0.00, type=float)  # memory_diversity损失比例
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
N_CLUSTERS = 8
FEATURE_TYPE = 'googlenet'
D_FEATURE = 1024  # for googlenet
# D_TXT_EMB = 300
D_C_OUTPUT = N_CLUSTERS  # category数量
MAX_COR = 0.0
GRAD_THRESHOLD = 10.0  # gradient threshold2

EVALUATE_MODEL = True
MIN_TRAIN_STEPS = 0
PRESTEPS = 0

if hp.server == 0:
    # path for JD server
    FEATURE_BASE = r'/public/data1/users/hulinkang/tvsum/tvsum_feature_googlenet_2fps/'
    SCORE_RECORD_PATH = r'/public/data1/users/hulinkang/tvsum/score_record_cluster.json'
    SEGINFO_PATH = r'/public/data1/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_segment_info.json'
    MODEL_SAVE_BASE = r'/public/data1/users/hulinkang/model_HL_Unify/'

elif hp.server == 1:
    # path for USTC servers
    FEATURE_BASE = r'/data/linkang/tvsum50/feature_googlenet_2fps/'
    SCORE_RECORD_PATH = r'/data/linkang/tvsum50/score_record_cluster.json'
    SEGINFO_PATH = r'/data/linkang/tvsum50/segment_info.json'
    MODEL_SAVE_BASE = r'/data/linkang/model_HL_v4/'

logging.basicConfig(level=logging.INFO)

def segment_embedding(feature):
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

def onehot(cluster_label):
    # 将list形式的cluster_label转化成onehot矩阵
    label = np.zeros((len(cluster_label), N_CLUSTERS))
    for i in range(len(cluster_label)):
        c = cluster_label[i]
        label[i, c] = 1
    return label

def load_data(feature_base, feature_type, score_record_path, seginfo_path):
    # 所有视频分为5份，每份中各个类别的视频各有一个
    with open(score_record_path, 'r') as file:
        score_record = json.load(file)
    with open(seginfo_path, 'r') as file:
        segment_info = json.load(file)

    vids = list(score_record.keys())
    # classify & sort
    video_split = {}
    for vid in vids:
        info = score_record[vid]
        if info['category'] not in video_split:
            video_split[info['category']] = []
        video_split[info['category']].append(vid)
    categories = list(video_split.keys())
    categories.sort()
    for c in categories:
        video_split[c].sort()
    # load & split data
    data = {}
    for i in range(5):
        data[i] = {}
        for j in range(len(categories)):  # 从各个类别的队首各取一个，保持顺序
            c = categories[j]
            vid = video_split[c].pop(0)
            data[i][vid] = {}

            # feature
            feature_path = feature_base + '%s_%s_2fps.npy' % (vid, feature_type)
            data[i][vid]['feature'] = np.load(feature_path)
            # label
            label_line = score_record[vid]['cluster_label']
            data[i][vid]['cluster_label'] = onehot(label_line)  # 用于对矩阵的每一列按照偏好计算loss
            data[i][vid]['rank_label'] = np.array(score_record[vid]['keyshot_label_uni'])  # 用于对合并后的得分计算loss
            # segment
            if hp.segment_num == 0:
                segments = np.zeros((0, D_FEATURE))
                poses = np.zeros((0))
            else:
                segments, poses = segment_embedding(data[i][vid]['feature'])
            data[i][vid]['segment_emb'] = segments
            data[i][vid]['segment_pos'] = poses
            data[i][vid]['category'] = score_record[vid]['category']

            logging.info('Vid: ' + str(vid) +
                         ' Feature: ' + str(data[i][vid]['feature'].shape) +
                         ' Cluster Label: ' + str(data[i][vid]['cluster_label'].shape) +
                         ' Rank Label: ' + str(data[i][vid]['rank_label'].shape) +
                         ' Segments: ' + str(segments.shape) +
                         ' Poses: ' + str(poses.shape)
                         )
    return data, score_record, segment_info

def find_pos_sample(score_record):
    # 考虑到每个视频中每种类型标签的数量并不平均，故在采样时围绕每种标签均建立若干个序列，保证对每种标签的序列内平衡性
    # 构建一个辅助字典，记录每个视频中每种标签所对应的正样本
    pos_sample_dict = {}
    for vid in score_record:
        labels = score_record[vid]['cluster_label']
        pos_sample_dict[vid] = {}
        for c in range(N_CLUSTERS):
            pos_sample_dict[vid][c] = []
        for p in range(len(labels)):
            pos_sample_dict[vid][labels[p]].append(p)  # 将这一帧的位置添加到对应标签的正样本列表中
    return pos_sample_dict

def train_scheme_build(data_train, pos_sample_dict, hp):
    # 输入拼接后的训练数据集
    # 按照固定比例从每个视频中随机抽取正负例
    # seqlen较长时可能会出现长度不足的序列，在getbatch中做padding
    train_scheme = []
    for vid in data_train:
        vlength = len(data_train[vid]['feature'])
        for c in range(N_CLUSTERS):  # 对每种标签都构建一组输入序列
            pos_list = pos_sample_dict[vid][c]
            neg_list = list(set(list(range(vlength))) - set(pos_list))
            pos_num = math.ceil(hp.seq_len * hp.pos_ratio)
            k = math.ceil(len(pos_list) / pos_num)  # 取正例的循环次数，不足时从头循环
            for i in range(k):
                pos_ind = i * pos_num % len(pos_list)
                neg_ind = i * (hp.seq_len - pos_num) % len(neg_list)
                frame_list = pos_list[pos_ind : pos_ind + pos_num]
                frame_list += neg_list[neg_ind : neg_ind + hp.seq_len - pos_num]
                frame_list.sort()
                train_scheme.append((vid, frame_list))
    random.shuffle(train_scheme)
    return train_scheme

def get_batch_train(data_train, train_scheme, step, hp):
    # 从train_scheme中获取gpu_num*bc个序列，每个长度seq_len，并返回每个frame的全局位置
    batch_num = hp.gpu_num * hp.bc
    features = []
    cluster_labels = []
    rank_labels = []
    positions = []
    segment_embs = []
    segment_poses = []
    scores = []
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(train_scheme)
        vid, frame_list = train_scheme[pos]
        vlength = len(data_train[str(vid)]['feature'])
        padding_len = hp.seq_len - len(frame_list)

        feature = data_train[vid]['feature'][frame_list]
        cluster_label = data_train[vid]['cluster_label'][frame_list]
        rank_label = data_train[vid]['rank_label'][frame_list]
        position = frame_list
        score =  np.ones((len(frame_list)))
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_FEATURE))
            cluster_label_pad = np.zeros((padding_len, D_C_OUTPUT))
            rank_label_pad = np.zeros((padding_len))
            position_pad = np.array([vlength] * padding_len)
            score_pad = np.zeros(padding_len)
            feature = np.vstack((feature, feature_pad))
            cluster_label = np.vstack((cluster_label, cluster_label_pad))
            rank_label = np.hstack((rank_label, rank_label_pad))
            position = np.hstack((position, position_pad))
            score = np.hstack((score, score_pad))

        features.append(feature)
        cluster_labels.append(cluster_label)
        rank_labels.append(rank_label)
        positions.append(position)
        scores.append(score)
        segment_embs.append(data_train[vid]['segment_emb'])
        segment_poses.append(data_train[vid]['segment_pos'])

    features = np.array(features)
    cluster_labels = np.array(cluster_labels)
    rank_labels = np.array(rank_labels)
    positions = np.array(positions)
    segment_embs = np.array(segment_embs)
    segment_poses = np.array(segment_poses)
    scores = np.array(scores)
    return features, positions, segment_embs, segment_poses, scores, cluster_labels, rank_labels

def test_scheme_build(data_test, hp):
    # 依次输入测试集中所有clip，不足seqlen的要补足，在getbatch中补足不够一个batch的部分
    # (vid, seq_start, seq_end)形式
    test_scheme = []
    test_vids = []
    for vid in data_test:
        vlength = len(data_test[str(vid)]['feature'])
        seq_num = math.ceil(vlength / hp.seq_len)
        for i in range(seq_num):
            test_scheme.append((vid, i * hp.seq_len, min(vlength, (i + 1) * hp.seq_len)))
        test_vids.append((vid, vlength))
    return test_scheme, test_vids

def get_batch_test(data_test, test_scheme, step, hp):
    # 输入拼接后的测试数据集
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
        vlength = len(data_test[str(vid)]['feature'])
        padding_len = hp.seq_len - (seq_end - seq_start)
        feature = data_test[str(vid)]['feature'][seq_start:seq_end]
        position = np.array(list(range(seq_start, seq_end)))
        score = np.ones(seq_end - seq_start)
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
        segment_embs.append(data_test[vid]['segment_emb'])
        segment_poses.append(data_test[vid]['segment_pos'])
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

def tower_loss_diverse(logits, cluster_labels, rank_labels, shots_output, memroy_output, hp):
    # logits: bc*seqlen*10
    # cluster_labels: bc*seqlen*10
    # rank_labels: bc*seqlen，用于对最大池化后的预测值计算损失
    # shots_output: bc*seqlen*D
    # memory_output: bc*memory_num*D

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

    # for cluster_labels: local prediction
    cluster_logits = tf.transpose(logits, perm=(0, 2, 1))  # bc*10*seq_len
    cluster_logits = tf.reshape(cluster_logits, shape=(-1, hp.seq_len))  # (bc*10)*seq_len
    cluster_labels_flat = tf.transpose(cluster_labels, perm=(0, 2, 1))
    cluster_labels_flat = tf.reshape(cluster_labels_flat, shape=(-1, hp.seq_len))

    cluster_nce_pos = tf.reduce_sum(tf.exp(cluster_labels_flat * cluster_logits), axis=1)
    cluster_nce_pos -= tf.reduce_sum((1 - cluster_labels_flat), axis=1)
    cluster_nce_all = tf.reduce_sum(tf.exp(cluster_logits), axis=1)
    cluster_nce_loss = -tf.log((cluster_nce_pos / cluster_nce_all) + 1e-5)
    cluster_loss = tf.reduce_mean(cluster_nce_loss)

    # for rank_labels: global prediction
    rank_logits = tf.reduce_max(logits, axis=2)  # bc*seqlen
    rank_nce_pos = tf.reduce_sum(tf.exp(rank_labels * rank_logits), axis=1)
    rank_nce_pos -= tf.reduce_sum((1 - rank_labels), axis=1)
    rank_nce_all = tf.reduce_sum(tf.exp(rank_logits), axis=1)
    rank_nce_loss = -tf.log((rank_nce_pos / rank_nce_all) + 1e-5)
    rank_loss = tf.reduce_mean(rank_nce_loss)

    # for shots diversity
    if hp.shots_div >= 0.01:
        top_50 = tf.nn.top_k(logits, int(hp.seq_len * hp.shots_div_ratio))
        top_indices = top_50.indices  # 没行（序列）前50%的索引
        KeyVecs = []  # 得分较高的shot对应的重建向量
        for i in range(top_indices.get_shape().as_list()[0]):  # i为展开后的pred第一维坐标
            seq_pos = int(i / D_C_OUTPUT)  # 对应的输出特征中的序列位置
            KeyVecs.append(tf.gather(shots_output[seq_pos:seq_pos + 1], top_indices[i], axis=1))
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

    loss = cluster_loss * hp.loss_cluster_ratio + \
           rank_loss * hp.loss_rank_ratio + \
           shots_diverse_loss * hp.shots_div + \
           mem_diverse_loss * hp.mem_div
    return loss, [cluster_loss, rank_loss, shots_diverse_loss, mem_diverse_loss]

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

def evaluation(pred_scores, test_videos, score_record):
    # 首先将模型输出裁剪为对每个视频中的帧得分预测
    preds_c = list(pred_scores[0])
    for i in range(1, len(pred_scores)):
        preds_c = preds_c + list(pred_scores[i])

    # 计算相关性系数
    predictions = {}
    pos = 0
    spearman_cors = []
    kendall_cors = []
    for vid, vlength in test_videos:
        y_pred = np.array(preds_c[pos:pos+vlength])  # vlength*10
        y_true = np.array(score_record[vid]['scores'])
        score_pred = np.max(y_pred, axis=1)  # 合并，取最大值
        predictions[vid] = y_pred
        pos += math.ceil(vlength / hp.seq_len) * hp.seq_len  # 必须跳过一个视频的末尾序列中padding的部分
        for i in range(len(y_true)):
            score_true = y_true[i]
            k_cor, p_value = stats.kendalltau(score_pred, score_true)
            s_cor, p_value = stats.spearmanr(score_pred, score_true)
            spearman_cors.append(s_cor)
            kendall_cors.append(k_cor)
    spearman_cors = np.array(spearman_cors)
    kendall_cors = np.array(kendall_cors)

    return np.mean(kendall_cors), np.mean(spearman_cors)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def model_clear(model_save_dir, max_cor):
    # 清除之前所有F1较小的模型
    models = []
    for name in os.listdir(model_save_dir):
        if name.endswith('.meta'):
            models.append(name.split('.meta')[0])
    for model in models:
        f1 = model.split('-')[-1]
        if f1.startswith('K') and float(f1.split('K')[-1]) < max_cor:
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

def run_training(data_train, data_test, score_record, segment_info, test_mode, model_save_dir, model_path):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    max_cor = MAX_COR

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        features_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_FEATURE))
        positions_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.seq_len))
        segment_embs_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.segment_num, D_FEATURE))
        segment_poses_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.segment_num))
        scores_src_holder = tf.placeholder(tf.float32,
                                           shape=(hp.bc * hp.gpu_num, hp.seq_len + hp.segment_num + hp.memory_num))
        cluster_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_C_OUTPUT))
        rank_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len))
        dropout_holder = tf.placeholder(tf.float32, shape=())
        training_holder = tf.placeholder(tf.bool, shape=())

        # memory initialization
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
        logits_list = []
        loss_list = []
        loss_ob_list = []
        for gpu_index in range(hp.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                features = features_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                positions = positions_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                segment_embs = segment_embs_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                segment_poses = segment_poses_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                scores_src = scores_src_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                cluster_labels = cluster_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                rank_labels = rank_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]

                # 整合concept与summary的预测，形成最终预测
                logits, shots_output, memory_output = transformer(segment_embs, features, memory_nodes,
                                                                  segment_poses, positions, scores_src,
                                                                  dropout_holder, training_holder, hp,
                                                                  D_C_OUTPUT)
                logits_list.append(logits)

                # 训练时每个序列只针对一个query预测summary
                loss, loss_ob = tower_loss_diverse(logits, cluster_labels, rank_labels, shots_output, memory_output, hp)
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
        if test_mode:
            logging.info(' Ckpt Model Restoring: ' + model_path)
            saver_overall.restore(sess, model_path)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        pos_sample_dict = find_pos_sample(score_record)
        train_scheme = train_scheme_build(data_train, pos_sample_dict, hp)
        test_scheme, test_vids = test_scheme_build(data_test, hp)
        epoch_step = math.ceil(len(train_scheme) / (hp.gpu_num * hp.bc))
        max_test_step = math.ceil(len(test_scheme) / (hp.gpu_num * hp.bc))

        # begin training
        ob_loss = []
        ob_sub_loss = []
        timepoint = time.time()
        for step in range(hp.maxstep):
            features_b, positions_b, segment_embs_b, segment_poses_b, scores_b, cluster_labels_b, rank_labels_b = \
                get_batch_train(data_train, train_scheme, step, hp)
            scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, hp.segment_num + hp.memory_num))))  # encoder中开放所有concept节点
            observe = sess.run([train_op] +
                               loss_list +
                               logits_list +
                               loss_ob_list,
                               feed_dict={features_holder: features_b,
                                          positions_holder: positions_b,
                                          scores_src_holder: scores_src_b,
                                          cluster_labels_holder: cluster_labels_b,
                                          rank_labels_holder: rank_labels_b,
                                          segment_embs_holder: segment_embs_b,
                                          segment_poses_holder: segment_poses_b,
                                          dropout_holder: hp.dropout,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1 + hp.gpu_num])
            sub_loss_batch = observe[-4:]
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均
            ob_sub_loss.append(sub_loss_batch)

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == hp.maxstep:
                if step == 0 and test_mode == 0:
                    continue
                train_scheme = train_scheme_build(data_train, pos_sample_dict, hp)  # shuffle train scheme
                duration = time.time() - timepoint
                timepoint = time.time()
                loss_array = np.array(ob_loss)
                ob_loss.clear()
                sub_loss_array = np.array(ob_sub_loss)
                sub_loss_array = np.mean(sub_loss_array, axis=0)
                ob_sub_loss.clear()
                logging.info(' Step %d: %.3f sec' % (step, duration))
                logging.info(' Evaluate: ' + str(step) + ' Epoch: ' + str(epoch))
                # logging.info(' Average Loss: ' + str(np.mean(loss_array)) + ' Min Loss: ' + str(
                #     np.min(loss_array)) + ' Max Loss: ' + str(np.max(loss_array)))
                logging.info('Total_Loss: %.4f Cluster_Loss: %.4f Rank_Loss: %.4f Shots_Diverse_Loss: %.4f Memory_Diverse_Loss: %.4f'%
                             (np.mean(loss_array), sub_loss_array[0], sub_loss_array[1], sub_loss_array[2], sub_loss_array[3]))
                if step < hp.protection or not int(epoch) % hp.eval_epoch == 0:
                    continue  # 增大测试间隔
                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                pred_scores = []
                for test_step in range(max_test_step):
                    features_b, positions_b, segment_embs_b, segment_poses_b, scores_b = \
                        get_batch_test(data_test, test_scheme, test_step, hp)
                    # print(test_step, test_scheme[test_step], features_b.mean(), end=' ')
                    scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, hp.segment_num + hp.memory_num))))  # encoder中开放所有concept节点
                    temp_list = sess.run(logits_list,feed_dict={features_holder: features_b,
                                                                        positions_holder: positions_b,
                                                                        scores_src_holder: scores_src_b,
                                                                        segment_embs_holder: segment_embs_b,
                                                                        segment_poses_holder: segment_poses_b,
                                                                        dropout_holder: hp.dropout,
                                                                        training_holder: False})
                    # print(np.array(temp_list).mean())
                    for preds in temp_list:
                        pred_scores.append(preds.reshape((-1, D_C_OUTPUT)))

                # p, r, f = evaluation(pred_scores, queries, query_summary, Tags, test_vids, concepts)
                k_cor, s_cor = evaluation(pred_scores,test_vids, score_record)
                logging.info('Kendall: %.3f, Spearman: %.3f' % (k_cor, s_cor))

                if test_mode == 1:
                    return k_cor
                # save model
                if step > MIN_TRAIN_STEPS - PRESTEPS and k_cor >= max_cor:
                    max_cor = k_cor
                    model_clear(model_save_dir, max_cor)
                    model_path = model_save_dir + 'S%d-E%d-L%.6f-K%.3f' % (step, epoch, np.mean(loss_array), k_cor)
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
    data, score_record, segment_info = load_data(FEATURE_BASE, FEATURE_TYPE, SCORE_RECORD_PATH, SEGINFO_PATH)

    # kfold
    model_scores = {}
    kfold_start = int(int(hp.start) / 10)
    repeat_start = int(hp.start) % 10
    for k in range(5):
        if k < kfold_start:
            continue
        train_split = [(k+0)%5, (k+1)%5, (k+2)%5]
        valid_split = [(k+3)%5]
        test_split = [(k+4)%5]
        data_train = {}
        data_valid = {}
        data_test = {}
        for i in train_split:
            data_train.update(data[i])
        for i in valid_split:
            data_valid.update(data[i])
        for i in test_split:
            data_test.update(data[i])

        # info
        logging.info('*' * 20 + 'Settings' + '*' * 20)
        logging.info('K-fold: ' + str(k))
        logging.info('Train: %s' % str(train_split))
        logging.info('Valid: %s  Test: %s' % (str(valid_split), str(test_split)))
        logging.info('Model Base: ' + MODEL_SAVE_BASE + hp.msd + '_%d' % k)
        logging.info('WarmUp: ' + str(hp.warmup))
        logging.info('Noam LR: ' + str(hp.lr_noam))
        logging.info('Num Heads: ' + str(hp.num_heads))
        logging.info('Num Blocks: ' + str(hp.num_blocks))
        logging.info('Batchsize: ' + str(hp.bc))
        logging.info('Max Steps: ' + str(hp.maxstep))
        logging.info('Dropout Rate: ' + str(hp.dropout))
        logging.info('Sequence Length: ' + str(hp.seq_len))
        logging.info('Evaluation Epoch: ' + str(hp.eval_epoch))
        logging.info('Positive Ratio: ' + str(hp.pos_ratio))
        logging.info('*' * 50)

        # repeat
        f1_scores = []
        for i in range(hp.repeat):
            if k == kfold_start and i < repeat_start:
                continue
            model_save_dir = MODEL_SAVE_BASE + hp.msd + '_%d_%d/' % (k, i)
            logging.info('*' * 10 + str(i) + ': ' + model_save_dir + '*' * 10)
            logging.info('*' * 60)
            run_training(data_train, data_valid, score_record, segment_info, 0, model_save_dir, '')
            logging.info('*' * 60)
            # model evaluation
            if EVALUATE_MODEL:
                models_to_restore = model_search(model_save_dir)
                for model_path in models_to_restore:
                    f1 = run_training(data_train, data_test, score_record, segment_info, 1, model_save_dir, model_path)
                    f1_scores.append(f1)
            if len(hp.end) > 0:
                kfold_end = int(int(hp.end) / 10)
                repeat_end = int(hp.end) % 10
                if k >= kfold_end and i >= repeat_end:
                    return
        logging.info('^' * 60 + '\n')
        model_scores[k] = f1_scores

    # print evluation
    if not EVALUATE_MODEL:
        return
    logging.info('*' * 20 + 'Results: ' + '*' * 20)
    scores_all = 0
    for k in model_scores:
        scores = model_scores[k]
        logging.info('Kfold: %s, Mean: %.3f, Scores: %s' %
                     (k, np.array(scores).mean(), str(scores)))
        scores_all += np.array(scores).mean()
    logging.info('Overall Results: %.3f' % (scores_all / 5))

if __name__ == '__main__':
    tf.app.run()

