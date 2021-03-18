# 在utc数据集上的实验，读取每个视频的shot（frame）特征，构建序列，取序列中间作为采样点，以固定比例的正负例训练
# 测试时取2%左右的输出标记为summary，计算基于二分图匹配的F1
# 用于query-focused，在训练时每个序列和所有query的组合都需要加入train_scheme，测试时依次对所有视频的所有query都进行一次计算
import os
import time
import numpy as np
import tensorflow as tf
import math
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import random
import logging
import argparse
import scipy.io
import h5py
import pickle
from SelfAttention_query_v0 import self_attention
from SelfAttention_query_v0 import D_MODEL
import networkx as nx


class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0',type=str)
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=2,type=int)
    parser.add_argument('--seq_len',default=11,type=int)
    parser.add_argument('--bc',default=10,type=int)
    parser.add_argument('--dropout',default='0.1',type=float)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--msd', default='utc_SA', type=str)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--lr_noam', default=2e-5, type=float)
    parser.add_argument('--warmup', default=6000, type=int)
    parser.add_argument('--maxstep', default=90000, type=int)
    parser.add_argument('--pos_ratio',default=0.5, type=float)
    parser.add_argument('--multimask',default=1, type=int)
    parser.add_argument('--kfold',default=3,type=int)
    parser.add_argument('--repeat',default=1,type=int)
    parser.add_argument('--dataset',default='utc',type=str)

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
REPEAT_TIMES = hp.repeat  # 重复训练和测试的次数
K_FOLD_MODE = hp.kfold  # 1-4，使用不同的集合划分

PRESTEPS = 0
WARMUP_STEP = hp.warmup
LR_NOAM = hp.lr_noam
MIN_TRAIN_STEPS = 4500
MAXSTEPS = hp.maxstep
PHASES_STEPS = [3000]
PHASES_LR = [4e-6, 1e-6]
HIDDEN_SIZE = 128  # for lstm
DROP_OUT = hp.dropout

EVL_EPOCHS = 1  # epochs for evaluation
L2_LAMBDA = 0.005  # weightdecay loss
GRAD_THRESHOLD = 10.0  # gradient threshold
MAX_F1 = 0.2

GPU_NUM = hp.gpu_num
BATCH_SIZE = hp.bc
SEQ_LEN = hp.seq_len
NUM_BLOCKS = hp.num_blocks
NUM_HEADS = hp.num_heads
MUlTIHEAD_ATTEN = hp.multimask
RECEP_SCOPES = 2#list(range(64))  # 用于multihead mask 从取样位置开始向两侧取的样本数量（单侧）

D_INPUT = 2048
D_CONCEPT_EMB = 300
POS_RATIO = hp.pos_ratio  # batch中正样本比例上限

load_ckpt_model = False

if hp.server == 0:
    # path for USTC server
    FEATURE_BASE = r'/public/data0/users/hulinkang/utc/features/'
    LABEL_BASE = r'/public/data0/users/hulinkang/utc/origin_data/Global_Summaries/'
    QUERY_SUM_BASE = r'/public/data0/users/hulinkang/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    EMBEDDING_PATH = r'/public/data0/users/hulinkang/utc/processed/query_dictionary.pkl'
    TAGS_PATH = r'/public/data0/users/hulinkang/utc/Tags.mat'
    model_save_base = r'/public/data0/users/hulinkang/model_HL_v4/'
    ckpt_model_path = r'/public/data0/users/hulinkang/model_HL_v4/utc_SA/'
else:
    # path for USTC servers
    FEATURE_BASE = r'/data/linkang/VHL_GNN/utc/features/'
    LABEL_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Global_Summaries/'
    QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    EMBEDDING_PATH = r'/data/linkang/VHL_GNN/utc/processed/query_dictionary.pkl'
    TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
    model_save_base = r'/data/linkang/model_HL_v4/'
    ckpt_model_path = r'/data/linkang/model_HL_v4/utc_SA/'

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

def load_feature_4fold(feature_base, label_base, Tags):
    data = {}
    for vid in range(1,5):
        data[str(vid)] = {}
        vlength = len(Tags[vid-1])
        # feature
        feature_path = feature_base + 'V%d_resnet_avg.h5' % vid
        label_path = label_base + 'P0%d/oracle.txt' % vid
        f = h5py.File(feature_path, 'r')
        feature = f['feature'][()][:vlength]
        data[str(vid)]['feature'] = feature
        # label
        with open(label_path, 'r') as f:
            hl = []
            for line in f.readlines():
                hl.append(int(line.strip())-1)
        label = np.zeros(len(feature))
        label[hl] = 1
        data[str(vid)]['labels'] = label
        # other
        data[str(vid)]['scores_avg'] = label + 1e-6  # just for padding
        data[str(vid)]['pos_index'] = np.where(label > 0)[0]
        data[str(vid)]['neg_index'] = np.where(label < 1)[0]
        logging.info('Vid: '+str(vid)+' Feature: '+str(feature.shape)+' Label: '+str(label.shape))
    return data

def query_process(query):
    # 对query进行处理
    transfer = {"Cupglass": "Glass",
                "Musicalinstrument": "Instrument",
                "Petsanimal": "Animal"}
    for i in range(len(query)):
        if query[i] in transfer:
            query[i] = transfer[query[i]]  # 单词置换
    query.sort()  # 按字典序组合两个词作为key
    return query

def load_query_summary(query_sum_base):
    # 加载query-focused oracle summary
    summary = {}
    for i in range(1,5):
        summary[str(i)] = {}
        summary_dir = query_sum_base + 'P0%d/' % i
        for root, dirs, files in os.walk(summary_dir):
            for file in files:
                concepts = query_process(file[:file.find("_oracle.txt")].split('_'))  # 提取query单词，置换并排序
                hl_shots = []
                with open(os.path.join(root,file),'r') as f:
                    for line in f.readlines():
                        hl_shots.append(int(line.strip())-1)
                summary[str(i)]['_'.join(concepts)] = hl_shots
    return summary

def train_scheme_build_v4(data_train,query_summary, seq_len):
    # 加入了query的train_scheme，在生成序列时匹配对应的concept，以及对应的标签
    # getbatch时获取concept对应的嵌入
    # 各个视频、各个query对应的序列全部打乱顺序训练，一个batch中可能包含来自不同视频、对应不同query的序列和标签
    pos_list = []
    neg_list  =[]
    for vid in data_train:
        for query in query_summary[vid]:
            hl_shot = np.array(query_summary[vid][query])
            vlength = len(data_train[vid]['scores_avg'])
            label = np.zeros(vlength)
            label[hl_shot] = 1
            pos_index = np.where(label > 0)[0]
            neg_index = np.where(label < 1)[0]
            # 遍历正样本索引与负样本索引中的所有样本，计算其邻域索引范围，分别加入两个列表
            for sample_pos in pos_index:
                seq_start = sample_pos - int(seq_len / 2)
                seq_end = seq_start + seq_len
                seq_start = max(0, seq_start)  # 截断
                pos_list.append((vid, query, seq_start, seq_end, sample_pos, 1))
            for sample_pos in neg_index:
                seq_start = sample_pos - int(seq_len / 2)
                seq_end = seq_start + seq_len
                seq_start = max(0, seq_start)  # 截断
                neg_list.append((vid, query, seq_start, seq_end, sample_pos, 0))
    random.shuffle(pos_list)
    random.shuffle(neg_list)
    return (pos_list, neg_list)

def get_batch_train_v3(data,query_summary,concept_embedding,train_scheme,step,gpu_num,bc,seq_len):
    # 可以调整正负样本比例的版本，不能使用pairwise loss
    # 加入了query，需要获取concept对应的嵌入
    pos_list,neg_list = train_scheme
    pos_num = len(pos_list)
    neg_num = len(neg_list)

    # 先生成batch_index并随机排列，然后按打乱后的顺序抽取sample_poses
    batch_index_raw = []
    pos_num_batch = math.ceil(gpu_num * bc * POS_RATIO)  # batch中正样本的数量
    neg_num_batch = gpu_num * bc - pos_num_batch
    for i in range(pos_num_batch):
        pos_position = (step * pos_num_batch + i) % pos_num
        batch_index_raw.append(pos_list[pos_position])
    for i in range(neg_num_batch):
        neg_position = (step * neg_num_batch + i) % neg_num
        batch_index_raw.append(neg_list[neg_position])
    random.shuffle(batch_index_raw)
    batch_index = []
    sample_poses = []
    batch_labels = []  # only for check
    for i in range(len(batch_index_raw)):
        vid, query, seq_start, seq_end, sample_pos, sample_label = batch_index_raw[i]
        batch_index.append((vid, query, seq_start, seq_end, sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)

    # 根据索引读取数据，并做padding
    features = []
    scores_avgs = []
    labels = []
    concept1_embs = []
    concept2_embs = []
    for i in range(len(batch_index)):
        vid,query,seq_start,seq_end,sample_pos = batch_index[i]
        c1,c2 = query.split('_')
        concept1_embs.append(concept_embedding[c1])
        concept2_embs.append(concept_embedding[c2])
        vlength = len(data[vid]['scores_avg'])
        seq_end = min(vlength,seq_end)  # 截断
        padding_len = seq_len - (seq_end - seq_start)
        feature = data[vid]['feature'][seq_start:seq_end]
        scores_avg = data[vid]['scores_avg'][seq_start:seq_end]
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_INPUT))
            scores_avg_pad = np.zeros((padding_len,))
            feature = np.vstack((feature,feature_pad))  # 统一在后侧padding
            scores_avg = np.hstack((scores_avg, scores_avg_pad))
        features.append(feature)
        scores_avgs.append(scores_avg)
        if sample_pos in query_summary[vid][query]:
            labels.append(1)
        else:
            labels.append(0)
    concept1_embs = np.array(concept1_embs).reshape((gpu_num * bc, D_CONCEPT_EMB))
    concept2_embs = np.array(concept2_embs).reshape((gpu_num * bc, D_CONCEPT_EMB))
    features = np.array(features).reshape((gpu_num * bc, seq_len, D_INPUT))
    scores_avgs = np.array(scores_avgs).reshape((gpu_num * bc, seq_len))
    labels = np.array(labels).reshape((gpu_num * bc,))
    sample_poses = np.array(sample_poses).reshape((gpu_num * bc,))

    # multihead mask
    h = NUM_HEADS
    mask_ranges = RECEP_SCOPES
    mask = np.ones((h,gpu_num*bc,seq_len))
    for i in range(h):
        # 对于每一个head，用一个感受范围做一组mask
        for j in range(gpu_num * bc):
            start = max(0, sample_poses[j] - mask_ranges)
            end = min(sample_poses[j] + mask_ranges + 1, seq_len)
            mask[i,j,start:end] = 0  # 第i个head第j个序列中的某一部分开放计算
    if not MUlTIHEAD_ATTEN:
        mask = np.zeros_like(mask)

    # check
    if np.sum(labels - np.array(batch_labels)) != 0:
        logging.info('Label Mismatch: %d' % step)
    return features, scores_avgs, labels, sample_poses, mask, concept1_embs, concept2_embs

def test_scheme_build_v4(data_test,query_summary,seq_len):
    # 加入query，对每个视频，依次测试每个query，需要记录vid以及query的顺序
    seq_list = []
    test_vids = []
    for vid in data_test:
        for query in query_summary[vid]:
            hl_shot = np.array(query_summary[vid][query])
            vlength = len(data_test[vid]['scores_avg'])
            label = np.zeros(vlength)
            label[hl_shot] = 1
            # 顺序将每个片段的邻域加入列表中，记录片段在序列中的位置以及片段标签
            for sample_pos in range(vlength):
                seq_start = sample_pos - int(seq_len / 2)
                seq_end = seq_start + seq_len
                seq_start = max(0, seq_start)  # 截断
                seq_list.append((vid, query, seq_start, seq_end, sample_pos, label[sample_pos]))
            test_vids.append((vid,query))  # 记录vid顺序用于evaluation
    return seq_list, test_vids

def get_batch_test_v3(data,query_summary,concept_embedding,test_scheme,step,gpu_num,bc,seq_len):
    # 与get_batch_test一致，每次选择gpu_num*bc个序列返回，但是保持原有顺序
    seq_list = test_scheme

    # 生成batch_index与sample_pos
    batch_index = []
    sample_poses = []  # 取样点在序列中的相对位置
    batch_labels = []  # only for check
    for i in range(gpu_num * bc):  # 每次预测gpu_num*bc个片段
        position = (step * gpu_num * bc + i) % len(seq_list)  # 当前起始位置，经过最后一个视频末尾后折返，多余的序列作为padding
        # 读取样本
        vid,query,seq_start,seq_end,sample_pos,sample_label = seq_list[position]
        batch_index.append((vid,query,seq_start,seq_end,sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)

    # 根据索引读取数据，并做padding
    features = []
    scores_avgs = []
    labels = []
    concept1_embs = []
    concept2_embs = []
    for i in range(len(batch_index)):
        vid,query,seq_start,seq_end,sample_pos = batch_index[i]
        c1,c2 = query.split('_')
        concept1_embs.append(concept_embedding[c1])
        concept2_embs.append(concept_embedding[c2])
        vlength = len(data[vid]['scores_avg'])
        seq_end = min(vlength,seq_end)  # 截断
        padding_len = seq_len - (seq_end - seq_start)
        feature = data[vid]['feature'][seq_start:seq_end]
        scores_avg = data[vid]['scores_avg'][seq_start:seq_end]
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_INPUT))
            scores_avg_pad = np.zeros((padding_len,))
            feature = np.vstack((feature,feature_pad))  # 统一在后侧padding
            scores_avg = np.hstack((scores_avg, scores_avg_pad))
        features.append(feature)
        scores_avgs.append(scores_avg)
        if sample_pos in query_summary[vid][query]:
            labels.append(1)
        else:
            labels.append(0)
    concept1_embs = np.array(concept1_embs).reshape((gpu_num * bc, D_CONCEPT_EMB))
    concept2_embs = np.array(concept2_embs).reshape((gpu_num * bc, D_CONCEPT_EMB))
    features = np.array(features).reshape((gpu_num * bc, seq_len, D_INPUT))
    scores_avgs = np.array(scores_avgs).reshape((gpu_num * bc, seq_len))
    labels = np.array(labels).reshape((gpu_num * bc,))
    sample_poses = np.array(sample_poses).reshape((gpu_num * bc,))

    # multihead mask
    h = NUM_HEADS
    mask_ranges = RECEP_SCOPES
    mask = np.ones((h,gpu_num*bc,seq_len))
    for i in range(h):
        # 对于每一个head，用一个感受范围做一组mask
        for j in range(gpu_num * bc):
            start = max(0, sample_poses[j] - mask_ranges)
            end = min(sample_poses[j] + mask_ranges + 1, seq_len)
            mask[i,j,start:end] = 0  # 第i个head第j个序列中的某一部分开放计算
    if not MUlTIHEAD_ATTEN:
        mask = np.zeros_like(mask)

    # check
    if np.sum(labels - np.array(batch_labels)) != 0:
        logging.info('Label Mismatch: %d' % step)
    return features, scores_avgs, labels, sample_poses, mask, concept1_embs, concept2_embs

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

def conv3d(name, l_input, w, b):
    return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def query_score_pred(features,scores_avg,sample_poses,multihead_mask,concept1,concept2,drop_out,training):
    # self-attention
    # feature形式为bc*seq_len个帧
    # 对encoder来说每个gpu上输入bc*seq_len*d，即每次输入bc个序列，每个序列长seq_len，每个元素维度为d
    # 在encoder中将输入的序列映射到合适的维度
    # 在训练过程中，依次计算两个concept上的得分，分别计算loss，然后求和作为这一batch的loss
    # 在测试过程中，依次计算两个concept上的得分，求和后作为整体得分，筛选hl
    seq_input = tf.reshape(features, shape=(BATCH_SIZE, SEQ_LEN, -1))  # bc*seq_len*1024
    multihead_mask = tf.reshape(multihead_mask, shape=(NUM_HEADS * BATCH_SIZE, SEQ_LEN))
    logits1, attention_list1 = self_attention(seq_input, scores_avg, None, multihead_mask, concept1,
                                            BATCH_SIZE, SEQ_LEN, NUM_BLOCKS, NUM_HEADS, drop_out, training)  # bc*seq_len
    logits2, attention_list2 = self_attention(seq_input, scores_avg, None, multihead_mask, concept2,
                                            BATCH_SIZE, SEQ_LEN, NUM_BLOCKS, NUM_HEADS, drop_out,
                                            training)  # bc*seq_len

    target = tf.one_hot(indices=sample_poses, depth=logits1.get_shape().as_list()[-1], on_value=1, off_value=0)
    target = tf.cast(target, dtype=tf.float32)
    logits1 = tf.reduce_sum(logits1 * target, axis=1)  # 只保留取样位置的值
    logits1 = tf.reshape(logits1, [-1, 1])
    logits2 = tf.reduce_sum(logits2 * target, axis=1)  # 只保留取样位置的值
    logits2 = tf.reshape(logits2, [-1, 1])

    logits1 = tf.clip_by_value(tf.reshape(tf.sigmoid(logits1), [-1, 1]), 1e-6, 0.999999)  # (bc*seq_len,1)
    logits2 = tf.clip_by_value(tf.reshape(tf.sigmoid(logits2), [-1, 1]), 1e-6, 0.999999)  # (bc*seq_len,1)

    return logits1, logits2, attention_list1, attention_list2

def _loss(sp,sn,delta):
    zeros = tf.constant(0,tf.float32,shape=[sp.get_shape().as_list()[0],1])
    delta_tensor = tf.constant(delta,tf.float32,shape=[sp.get_shape().as_list()[0],1])
    u = 1 - sp + sn
    lp = tf.maximum(zeros,u)
    condition = tf.less(u,delta_tensor)
    v = tf.square(lp)*0.5
    w = lp*delta-delta*delta*0.5
    loss = tf.where(condition,x=v,y=w)
    return tf.reduce_mean(loss)

def tower_loss_huber(name_scope,preds,labels):
    # 每一组相邻的分段计算一次loss，取平均
    cij_list = []
    for i in range(BATCH_SIZE - 1):
        condition = tf.greater(labels[i],labels[i+1])
        sp = tf.where(condition,preds[i],preds[i+1])
        sn = tf.where(condition,preds[i+1],preds[i])
        cij = _loss(sp,sn,3)
        cij_list.append(cij)
    cost = cij_list[0]
    for i in range(1,len(cij_list)):
        cost = cost + cij_list[i]
    cost = cost / len(cij_list)
    # weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))
    # total_loss = cost + weight_decay_loss
    total_loss = cost

    return tf.reduce_mean(total_loss)

def tower_loss(name_scope,logits,labels):
    y = tf.reshape(labels,[-1,1])
    ce = -y * (tf.log(logits)) - (1 - y) * tf.log(1 - logits)
    loss = tf.reduce_mean(ce)
    return loss

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

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

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

def evaluation_query(pred_scores, data_test, query_summary, test_vids, Tags):
    # 对每个视频的每个query对应的summary进行
    # 读入全部预测结果，按照视频切分，以一定的比例选出前几个shot，进行匹配并计算F1
    # 首先将模型输出裁剪为对每个视频中的帧得分预测
    preds_c = list(pred_scores[0])
    for i in range(1, len(pred_scores)):
        preds_c = preds_c + list(pred_scores[i])
    y_preds = {}
    # 计算F1
    pos = 0
    PRE_values = []
    REC_values = []
    F1_values = []
    for vid,query in test_vids:
        hl_shot = query_summary[vid][query]
        label = np.zeros(len(data_test[vid]['scores_avg']))
        label[hl_shot] = 1
        vlength = len(label)
        y_pred = np.array(preds_c[pos:pos + vlength])
        y_preds[vid] = y_pred
        pos += vlength
        y_pred_list = list(y_pred)
        y_pred_list.sort(reverse=True)
        threshold = y_pred_list[math.ceil(vlength * 0.02)]
        shot_seq_pred = np.where(y_pred >= threshold)[0]
        shot_seq_label = np.where(label > 0)[0]
        # matching
        sim_mat = similarity_compute(Tags, int(vid), shot_seq_pred, shot_seq_label)
        weight = shot_matching(sim_mat)
        precision = weight / len(shot_seq_pred)
        recall = weight / len(shot_seq_label)
        f1 = 2 * precision * recall / (precision + recall)
        PRE_values.append(precision)
        REC_values.append(recall)
        F1_values.append(f1)
    PRE_values = np.array(PRE_values)
    REC_values = np.array(REC_values)
    F1_values = np.array(F1_values)
    return np.mean(PRE_values), np.mean(REC_values), np.mean(F1_values)

def run_training(data_train, data_test, Tags, query_summary, concept_embedding, test_mode, model_save_dir):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    max_f1 = MAX_F1

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        features_holder = tf.placeholder(tf.float32, shape=(BATCH_SIZE * GPU_NUM, SEQ_LEN, D_INPUT))
        scores_holder = tf.placeholder(tf.float32, shape=(BATCH_SIZE * GPU_NUM, SEQ_LEN))
        labels_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,))
        sample_poses_holder = tf.placeholder(tf.int32,shape=(BATCH_SIZE * GPU_NUM,))
        mask_holder = tf.placeholder(tf.float32,shape=(NUM_HEADS, BATCH_SIZE * GPU_NUM, SEQ_LEN))
        concept1_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,D_CONCEPT_EMB))
        concept2_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,D_CONCEPT_EMB))
        dropout_holder = tf.placeholder(tf.float32,shape=())
        training_holder = tf.placeholder(tf.bool,shape=())

        # training operations
        lr = noam_scheme(LR_NOAM,global_step,WARMUP_STEP)
        # lr = tf.train.piecewise_constant(global_step,PHASES_STEPS,PHASES_LR)
        opt_train = tf.train.AdamOptimizer(lr)

        # graph building
        tower_grads_train = []
        logits_list = []
        loss_list = []
        attention_list1 = []
        attention_list2 = []
        for gpu_index in range(GPU_NUM):
            with tf.device('/gpu:%d' % gpu_index):
                features = features_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                labels = labels_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,]
                scores_avg = scores_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                sample_poses = sample_poses_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,]
                concept1 = concept1_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                concept2 = concept2_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                multihead_mask = mask_holder[:, gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]  # 从bc维切割

                # predict scores
                logits1, logits2, atlist_one1, atlist_one2 = query_score_pred(features, scores_avg, sample_poses, multihead_mask,
                                                concept1, concept2, dropout_holder, training_holder)
                logits = (logits1 + logits2) / 2
                logits_list.append(logits)
                attention_list1 += atlist_one1  # 逐个拼接各个卡上的attention_list
                attention_list2 += atlist_one2
                # calculate loss & gradients
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                # loss = tower_loss_huber(loss_name_scope, logits, labels)
                loss1 = tower_loss(loss_name_scope,logits1,labels)
                loss2 = tower_loss(loss_name_scope,logits2,labels)
                loss = (loss1 + loss2) / 2
                varlist = tf.trainable_variables()  # 全部训练
                grads_train = opt_train.compute_gradients(loss, varlist)
                thresh = GRAD_THRESHOLD  # 梯度截断 防止爆炸
                grads_train_cap = [(tf.clip_by_value(grad, -thresh, thresh), var) for grad, var in grads_train]
                tower_grads_train.append(grads_train_cap)
                loss_list.append(loss)
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
        if load_ckpt_model:
            logging.info(' Ckpt Model Restoring: '+ckpt_model_path)
            saver_overall.restore(sess, ckpt_model_path)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        train_scheme = train_scheme_build_v4(data_train, query_summary, SEQ_LEN)
        # epoch_step = math.ceil(len(train_scheme[0]) / (BATCH_SIZE * GPU_NUM / 2))  # 所有正样本都计算过一次作为一个epoch
        # 正负样本比例可变的版本
        epoch_step = math.ceil(len(train_scheme[0]) / (BATCH_SIZE * GPU_NUM * POS_RATIO)) # 所有正样本都计算过一次作为一个epoch
        # epoch_step = 100

        test_scheme, test_vids = test_scheme_build_v4(data_test, query_summary, SEQ_LEN)
        max_test_step = math.ceil(len(test_scheme) / BATCH_SIZE / GPU_NUM)

        # Beging training
        ob_loss = []
        timepoint = time.time()
        for step in range(MAXSTEPS):
            features_b, scores_avg_b, labels_b, sample_poses_b, mask_b, concept1, concept2 = get_batch_train_v3(
                data_train, query_summary, concept_embedding, train_scheme, step,GPU_NUM,BATCH_SIZE,SEQ_LEN)
            observe = sess.run([train_op] + loss_list + logits_list + attention_list1 + [global_step, lr],
                               feed_dict={features_holder: features_b,
                                          scores_holder: scores_avg_b,
                                          labels_holder: labels_b,
                                          sample_poses_holder: sample_poses_b,
                                          mask_holder: mask_b,
                                          concept1_holder: concept1,
                                          concept2_holder: concept2,
                                          dropout_holder: DROP_OUT,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1+GPU_NUM])
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == MAXSTEPS:
                if step == 0 and test_mode == 0:
                    continue
                duration = time.time() - timepoint
                timepoint = time.time()
                loss_array = np.array(ob_loss)
                ob_loss.clear()
                logging.info(' Step %d: %.3f sec' % (step, duration))
                logging.info(' Evaluate: '+str(step)+' Epoch: '+str(epoch))
                logging.info(' Average Loss: '+str(np.mean(loss_array))+' Min Loss: '+str(np.min(loss_array))+' Max Loss: '+str(np.max(loss_array)))

                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                pred_scores = []  # 每个batch输出的预测得分
                for test_step in range(max_test_step):
                    features_b, scores_avg_b, labels_b, sample_poses_b, mask_b, concept1, concept2 = get_batch_test_v3(
                        data_test, query_summary, concept_embedding, test_scheme, test_step, GPU_NUM, BATCH_SIZE, SEQ_LEN)
                    logits_temp_list = sess.run(logits_list, feed_dict={features_holder: features_b,
                                                                        scores_holder: scores_avg_b,
                                                                        sample_poses_holder: sample_poses_b,
                                                                        mask_holder: mask_b,
                                                                        concept1_holder: concept1,
                                                                        concept2_holder: concept2,
                                                                        training_holder: False,
                                                                        dropout_holder: 0})
                    for preds in logits_temp_list:
                        pred_scores.append(preds.reshape((-1)))
                p, r, f = evaluation_query(pred_scores, data_test, query_summary, test_vids, Tags)
                logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f))
                pos_list, neg_list = train_scheme
                random.shuffle(pos_list)
                random.shuffle(neg_list)
                train_scheme = (pos_list, neg_list)

                if test_mode == 1:
                    return
                # save model
                if step > MIN_TRAIN_STEPS - PRESTEPS and f >= max_f1:
                    if f > max_f1:
                        max_f1 = f
                    model_path = model_save_dir + 'S%d-E%d-L%.6f-F%.3f' % (step,epoch,np.mean(loss_array),f)
                    saver_overall.save(sess, model_path)
                    logging.info('Model Saved: '+model_path+'\n')

            if step % 3000 == 0 and step > 0:
                model_path = model_save_dir + 'S%d-E%d' % (step+PRESTEPS, epoch)
                # saver_overall.save(sess, model_path)
                logging.info('Model Saved: '+str(step + PRESTEPS))

            # saving final model
        model_path = model_save_dir + 'S%d' % (MAXSTEPS + PRESTEPS)
        # saver_overall.save(sess, model_path)
        logging.info('Model Saved: '+str(MAXSTEPS + PRESTEPS))

    return

def main(self):
    # load data
    Tags = load_Tags(TAGS_PATH)
    data = load_feature_4fold(FEATURE_BASE, LABEL_BASE, Tags)
    query_summary = load_query_summary(QUERY_SUM_BASE)
    with open(EMBEDDING_PATH,'rb') as f:
        concept_embedding = pickle.load(f)

    # split data
    data_train = {}
    data_valid = {}
    data_test = {}
    data_train[str((K_FOLD_MODE+0) % 4 + 1)] = data[str((K_FOLD_MODE+0) % 4 + 1)]
    data_train[str((K_FOLD_MODE+1) % 4 + 1)] = data[str((K_FOLD_MODE+1) % 4 + 1)]
    data_valid[str((K_FOLD_MODE + 2) % 4 + 1)] = data[str((K_FOLD_MODE + 2) % 4 + 1)]
    data_test[str((K_FOLD_MODE + 3) % 4 + 1)] = data[str((K_FOLD_MODE + 3) % 4 + 1)]

    # info
    logging.info('*'*20+'Settings'+'*'*20)
    logging.info('K-fold: ' + str(K_FOLD_MODE))
    logging.info('Valid: %d  Test: %d' % ((K_FOLD_MODE + 2) % 4 + 1, (K_FOLD_MODE + 3) % 4 + 1))
    logging.info('Model Base: '+model_save_base+hp.msd)
    logging.info('Training Phases: ' + str(PHASES_STEPS))
    logging.info('Phase LR: '+str(PHASES_LR))
    logging.info('WarmUp: ' + str(WARMUP_STEP))
    logging.info('Noam LR: ' + str(LR_NOAM))
    logging.info('Num Heads: '+str(NUM_HEADS))
    logging.info('Num Blocks: '+str(NUM_BLOCKS))
    logging.info('Batchsize: '+str(BATCH_SIZE))
    logging.info('Max Steps: '+str(MAXSTEPS))
    logging.info('Dropout Rate: '+str(DROP_OUT))
    logging.info('Sequence Length: '+str(SEQ_LEN))
    logging.info('*' * 50+'\n')

    # repeat training for test
    for i in range(REPEAT_TIMES):
        model_save_dir = model_save_base + hp.msd + '_%d/' % i
        logging.info('*'*10+str(i)+': '+model_save_dir+'*'*10)
        logging.info('*'*60)
        run_training(data_train, data_valid, Tags, query_summary, concept_embedding, 0, model_save_dir)  # for training
        logging.info('*' * 60)

if __name__ == '__main__':
    tf.app.run()



