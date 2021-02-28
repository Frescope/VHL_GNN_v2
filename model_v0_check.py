# 新实验，实验1
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
import SelfAttention_v0
from SelfAttention_v0 import self_attention
from SelfAttention_v0 import D_MODEL
from tools.knapsack_iter import knapSack

class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='3',type=str)
    parser.add_argument('--num_heads',default=64,type=int)
    parser.add_argument('--num_blocks',default=4,type=int)
    parser.add_argument('--seq_len',default=70,type=int)
    parser.add_argument('--bc',default=10,type=int)
    parser.add_argument('--dropout',default='0.15',type=float)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--msd', default='tvsum_SA', type=str)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--lr_noam', default=1e-6, type=float)
    parser.add_argument('--warmup', default=6000, type=int)
    parser.add_argument('--maxstep', default=45000, type=int)
    parser.add_argument('--pos_ratio',default=0.8, type=float)
    parser.add_argument('--multimask',default=1, type=int)
    parser.add_argument('--kfold', default=0, type=int)
    parser.add_argument('--repeat', default=10, type=int)

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
K_FOLD_MODE = hp.kfold  # 0-4，使用不同的集合划分

PRESTEPS = 0
WARMUP_STEP = hp.warmup
LR_NOAM = hp.lr_noam
MIN_TRAIN_STEPS = 2000
MAXSTEPS = hp.maxstep
PHASES_STEPS = [3000]
PHASES_LR = [4e-6, 1e-6]
HIDDEN_SIZE = 128  # for lstm
DROP_OUT = hp.dropout

EVL_EPOCHS = 1  # epochs for evaluation
L2_LAMBDA = 0.005  # weightdecay loss
GRAD_THRESHOLD = 10.0  # gradient threshold
MAX_F1 = 0.6

GPU_NUM = hp.gpu_num
BATCH_SIZE = hp.bc
SEQ_LEN = hp.seq_len
NUM_BLOCKS = hp.num_blocks
NUM_HEADS = hp.num_heads
MUlTIHEAD_ATTEN = hp.multimask
RECEP_SCOPES = list(range(64))  # 用于multihead mask 从取样位置开始向两侧取的样本数量（单侧）

D_INPUT = 1024
POS_RATIO = hp.pos_ratio  # batch中正样本比例上限

load_ckpt_model = True

if hp.server == 0:
    # path for USTC server
    SCORE_PATH = r'/public/data0/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_score_record.json'
    VCAT_PATH = r'/public/data0/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_video_category.json'
    SEGINFO_PATH = r'/public/data0/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_segment_info.json'
    FEATURE_DIR = r'/public/data0/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_feature_googlenet_2fps/'
    model_save_base = r'/public/data0/users/hulinkang/model_HL_v2/'
    # ckpt_model_path = '../../model_HL_v4/tvsum_SA/STEP_5000'
    ckpt_model_path = '../../model_HL_v4/tvsum_SA/S20376-E24-L0.010669-F0.512'
else:
    # path for USTC servers
    SCORE_PATH = r'/data/linkang/VHL_GNN/tvsum_score_record.json'
    VCAT_PATH = r'/data/linkang/VHL_GNN/tvsum_video_category.json'
    SEGINFO_PATH = r'/data/linkang/VHL_GNN/tvsum_segment_info.json'
    FEATURE_DIR = r'/data/linkang/VHL_GNN/tvsum_feature_googlenet_2fps/'
    model_save_base = r'/data/linkang/model_HL_v4/'
    # ckpt_model_path = '../model_HL_v2/tvsum_SA/STEP_5000'
    ckpt_model_path = '../model_HL_v2/tvsum_SA/S20376-E24-L0.010669-F0.512'

logging.basicConfig(level=logging.INFO)

def load_info(score_path,vcat_path,seginfo_path):
    with open(score_path,'r') as file:
        score_record = json.load(file)
    with open(vcat_path,'r') as file:
        video_category = json.load(file)
    with open(seginfo_path,'r') as file:
        segment_info = json.load(file)
    return score_record, video_category, segment_info

def load_feature(score_record,video_category,feature_dir):
    # split dataset
    categories = list(video_category.keys())
    train_vids = []
    valid_vids = []
    test_vids = []
    for cate in categories:
        names = video_category[cate]
        names.sort()
        valid_vids.append(names.pop())
        test_vids.append(names.pop())
        train_vids += names

    # load data
    vids = list(score_record.keys())
    data_train = {}
    data_valid = {}
    data_test = {}
    for vid in vids:
        temp = {}
        temp['feature'] = np.load(feature_dir + vid +'_googlenet_2fps.npy')
        temp['scores'] = np.array(score_record[vid]['scores'])
        temp['scores_avg'] = np.array(score_record[vid]['scores_avg'])
        temp['labels'] = np.array(score_record[vid]['label_greedy'])
        temp['pos_index'] = np.where(temp['labels'] > 0)[0]
        temp['neg_index'] = np.where(temp['labels'] < 1)[0]
        if vid in train_vids:
            data_train[vid] = temp
        elif vid in valid_vids:
            data_valid[vid] = temp
        else:
            data_test[vid] = temp
        logging.info(vid+': '+str(temp['feature'].shape)+str(temp['scores'].shape)+str(temp['labels'].shape)+
                     ' pos_num: %d neg_num: %d' % (len(temp['pos_index']), len(temp['neg_index'])))
    logging.info('Valid Set: '+str(valid_vids))
    logging.info('Test Set: '+str(test_vids))
    return data_train, data_valid, data_test

def f1_calc(pred,gts):
    # 计算pred与所有gt的平均f1
    f1s = []
    for gt in gts:
        precision = np.sum(pred * gt) / (np.sum(pred) + 1e-6)
        recall = np.sum(pred * gt) / (np.sum(gt) + 1e-6)
        f1s.append(2 * precision * recall / (precision + recall + 1e-6))
    return np.array(f1s).mean()

def max_f1_estimate(score_record, vids):
    # 使用scores_avg作为预测，计算与各个summary的F1，作为对模型可能达到的最大F1的估计
    f1_overall_greedy = []
    for vid in vids:
        label_trues = score_record[vid]['keyshot_labels']
        label_greedy = np.array(score_record[vid]['label_greedy'])
        f1_greedy = f1_calc(label_greedy,label_trues)
        f1_overall_greedy.append(f1_greedy)
    return np.array(f1_overall_greedy)

def load_feature_5fold(score_record,video_category,feature_dir):
    # 将数据划分为5部分，提取特征，每个部分都由各个类别各取一个视频组成
    # split dataset
    categories = list(video_category.keys())
    subset_indexes = [[],[],[],[],[]]  # 5个列表，分别记录属于每个子集的索引
    for cate in categories:
        names = video_category[cate]
        names.sort()
        for i in range(len(names)):
            subset_indexes[i].append(names[i])

    # load data
    vids = list(score_record.keys())
    subsets = [{},{},{},{},{}]  # 5个字典，每个是一个数据子集
    for vid in vids:
        temp = {}
        temp['feature'] = np.load(feature_dir + vid +'_googlenet_2fps.npy')
        temp['scores'] = np.array(score_record[vid]['scores'])
        temp['scores_avg'] = np.array(score_record[vid]['scores_avg'])
        temp['labels'] = np.array(score_record[vid]['label_greedy'])
        temp['pos_index'] = np.where(temp['labels'] > 0)[0]
        temp['neg_index'] = np.where(temp['labels'] < 1)[0]
        for i in range(len(subset_indexes)):
            index = subset_indexes[i]
            if vid in index:
                subsets[i][vid] = temp
                break
            logging.info(
                vid + ': ' + str(temp['feature'].shape) + str(temp['scores'].shape) + str(temp['labels'].shape) +
                ' pos_num: %d neg_num: %d' % (len(temp['pos_index']), len(temp['neg_index'])))

    for i in range(len(subset_indexes)):
        logging.info('Subset '+str(i)+str(subset_indexes[i]))
        f1_subset = max_f1_estimate(score_record, subset_indexes[i])
        # logging.info('Estimated F1: '+str(list(f1_subset)))
        logging.info('Average Estimated F1: '+str(f1_subset.mean()))
    return subsets

def train_scheme_build_v3(data_train,seq_len):
    # 根据正负样本制定的train_scheme，取每个样本的左右领域与样本共同构成一个序列，分别得到正样本序列与负样本序列
    # 在getbatch时数据用零填充，score也用零填充，在attention计算时根据score将负无穷输入softmax，消除padding片段对有效片段的影响
    # 正负样本序列生成后随机化，直接根据step确定当前使用哪个序列，正负各取一个计算pairwise loss
    # train_scheme = [pos_list=(vid,seq_start,seq_end,sample_pos,sample_label),neg_list=()]

    pos_list = []
    neg_list = []
    for vid in data_train:
        label = data_train[vid]['labels']
        vlength = len(label)
        pos_index = data_train[vid]['pos_index']
        neg_index = data_train[vid]['neg_index']
        # 遍历正样本索引与负样本索引中的所有样本，计算其邻域索引范围，分别加入两个列表
        for sample_pos in pos_index:
            seq_start = sample_pos - int(seq_len / 2)
            seq_end = seq_start + seq_len
            seq_start = max(0, seq_start)  # 截断
            pos_list.append((vid,seq_start,seq_end,sample_pos,1))
        for sample_pos in neg_index:
            seq_start = sample_pos - int(seq_len / 2)
            seq_end = seq_start + seq_len
            seq_start = max(0, seq_start)  # 截断
            neg_list.append((vid,seq_start,seq_end,sample_pos,0))
    random.shuffle(pos_list)
    random.shuffle(neg_list)
    return (pos_list,neg_list)

def get_batch_train(data,train_scheme,step,gpu_num,bc,seq_len):
    # 按照train-scheme制作batch，每次选择gpu_num*bc个序列返回，要求每个bc中一半是正样本一半是负样本，交替排列
    # 每个序列构成一个sample，故共有gpu_num*bc个sample，每个gpu上计算bc个sample的loss
    # 返回gpu_num*bc个label，对应每个sample中一个片段的标签
    # 同时返回一个取样位置序列sample_pos，顺序记录每个sample中标签对应的片段在序列中的位置，模型输出后根据sample_pos计算loss
    # 根据step顺序读取pos_list与neg_list中的序列并组合为batch_index，再抽取对应的visual，audio，score与label
    # 产生multihead mask，(gpu_num*num_heads*bc) * seq_len的矩阵
    pos_list,neg_list = train_scheme
    pos_num = len(pos_list)
    neg_num = len(neg_list)

    # 生成batch_index与sample_pos
    batch_index = []
    sample_poses = []
    batch_labels = []  # only for check
    for i in range(int(gpu_num * bc / 2)):  # gpu_num*bc应当为偶数
        pos_position = (step * int(gpu_num * bc / 2) + i) % pos_num  # 当前在pos_list中的起始位置
        neg_position = (step * int(gpu_num * bc / 2) + i) % neg_num  # 当前在neg_list中的起始位置
        # 读正样本
        vid,seq_start,seq_end,sample_pos,sample_label = pos_list[pos_position]
        batch_index.append((vid,seq_start,seq_end,sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)
        # 读负样本
        vid, seq_start, seq_end, sample_pos, sample_label = neg_list[neg_position]
        batch_index.append((vid, seq_start, seq_end, sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)

    # 根据索引读取数据，并做padding
    features = []
    scores_avgs = []
    labels = []
    for i in range(len(batch_index)):
        vid,seq_start,seq_end,sample_pos = batch_index[i]
        vlength = len(data[vid]['labels'])
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
        labels.append(data[vid]['labels'][sample_pos])
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
            start = max(0, sample_poses[j] - mask_ranges[i])
            end = min(sample_poses[j] + mask_ranges[i] + 1, seq_len)
            mask[i,j,start:end] = 0  # 第i个head第j个序列中的某一部分开放计算
    if not MUlTIHEAD_ATTEN:
        mask = np.zeros_like(mask)

    # check
    if np.sum(labels - np.array(batch_labels)) != 0:
        logging.info('Label Mismatch: %d' % step)

    return features, scores_avgs, labels, sample_poses, mask

def get_batch_train_v2(data,train_scheme,step,gpu_num,bc,seq_len):
    # 可以调整正负样本比例的版本，不能使用pairwise loss
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
        vid, seq_start, seq_end, sample_pos, sample_label = batch_index_raw[i]
        batch_index.append((vid, seq_start, seq_end, sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)

    # 根据索引读取数据，并做padding
    features = []
    scores_avgs = []
    labels = []
    for i in range(len(batch_index)):
        vid,seq_start,seq_end,sample_pos = batch_index[i]
        vlength = len(data[vid]['labels'])
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
        labels.append(data[vid]['labels'][sample_pos])
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
            start = max(0, sample_poses[j] - mask_ranges[i])
            end = min(sample_poses[j] + mask_ranges[i] + 1, seq_len)
            mask[i,j,start:end] = 0  # 第i个head第j个序列中的某一部分开放计算
    if not MUlTIHEAD_ATTEN:
        mask = np.zeros_like(mask)

    # check
    if np.sum(labels - np.array(batch_labels)) != 0:
        logging.info('Label Mismatch: %d' % step)
    return features, scores_avgs, labels, sample_poses, mask

def test_scheme_build(data_test,seq_len):
    # 与train_schem_build一致，但是不区分正负样本，也不做随机化
    seq_list = []
    test_vids = []
    for vid in data_test:
        label = data_test[vid]['labels']
        vlength = len(label)
        # 顺序将每个片段的邻域加入列表中，记录片段在序列中的位置以及片段标签
        for sample_pos in range(vlength):
            seq_start = sample_pos - int(seq_len / 2)
            seq_end = seq_start + seq_len
            seq_start = max(0, seq_start)  # 截断
            seq_list.append((vid, seq_start, seq_end, sample_pos, label[sample_pos]))
        test_vids.append(vid)  # 记录vid顺序用于evaluation
    return seq_list, test_vids

def get_batch_test(data,test_scheme,step,gpu_num,bc,seq_len):
    # 与get_batch_test一致，每次选择gpu_num*bc个序列返回，但是保持原有顺序
    seq_list = test_scheme

    # 生成batch_index与sample_pos
    batch_index = []
    sample_poses = []  # 取样点在序列中的相对位置
    batch_labels = []  # only for check
    for i in range(gpu_num * bc):  # 每次预测gpu_num*bc个片段
        position = (step * gpu_num * bc + i) % len(seq_list)  # 当前起始位置，经过最后一个视频末尾后折返，多余的序列作为padding
        # 读取样本
        vid,seq_start,seq_end,sample_pos,sample_label = seq_list[position]
        batch_index.append((vid,seq_start,seq_end,sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)

    # 根据索引读取数据，并做padding
    features = []
    scores_avgs = []
    labels = []
    for i in range(len(batch_index)):
        vid, seq_start, seq_end, sample_pos = batch_index[i]
        vlength = len(data[vid]['labels'])
        seq_end = min(vlength, seq_end)  # 截断
        padding_len = seq_len - (seq_end - seq_start)
        feature = data[vid]['feature'][seq_start:seq_end]
        scores_avg = data[vid]['scores_avg'][seq_start:seq_end]
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_INPUT))
            scores_avg_pad = np.zeros((padding_len,))
            feature = np.vstack((feature, feature_pad))  # 统一在后侧padding
            scores_avg = np.hstack((scores_avg, scores_avg_pad))
        features.append(feature)
        scores_avgs.append(scores_avg)
        labels.append(data[vid]['labels'][sample_pos])
    features = np.array(features).reshape((gpu_num * bc, seq_len, D_INPUT))
    scores_avgs = np.array(scores_avgs).reshape((gpu_num * bc, seq_len))
    labels = np.array(labels).reshape((gpu_num * bc,))
    sample_poses = np.array(sample_poses).reshape((gpu_num * bc,))

    # multihead mask
    h = NUM_HEADS
    mask_ranges = RECEP_SCOPES
    mask = np.ones((h, gpu_num * bc, seq_len))
    for i in range(h):
        # 对于每一个head，用一个感受范围做一组mask
        for j in range(gpu_num * bc):
            start = max(0, sample_poses[j] - mask_ranges[i])
            end = min(sample_poses[j] + mask_ranges[i] + 1, seq_len)
            mask[i, j, start:end] = 0  # 第i个head第j个序列中的某一部分开放计算
    if not MUlTIHEAD_ATTEN:
        mask = np.zeros_like(mask)

    # check
    if np.sum(labels - np.array(batch_labels)) != 0:
        logging.info('Label Mismatch: %d' % step)
    return features, scores_avgs, labels, sample_poses, mask

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

def score_pred(features,scores_avg,sample_poses,multihead_mask,drop_out,training):

    # # self-attention
    # # feature形式为bc*seq_len个帧
    # # 对encoder来说每个gpu上输入bc*seq_len*d，即每次输入bc个序列，每个序列长seq_len，每个元素维度为d
    # # 在encoder中将输入的序列映射到合适的维度
    seq_input = tf.reshape(features, shape=(BATCH_SIZE, SEQ_LEN, -1))  # bc*seq_len*1024
    multihead_mask = tf.reshape(multihead_mask, shape=(NUM_HEADS * BATCH_SIZE, SEQ_LEN))
    logits, attention_list = self_attention(seq_input, scores_avg, multihead_mask, SEQ_LEN, NUM_BLOCKS,
                                            NUM_HEADS, drop_out, training)  # bc*seq_len

    target = tf.one_hot(indices=sample_poses, depth=logits.get_shape().as_list()[-1], on_value=1, off_value=0)
    target = tf.cast(target, dtype=tf.float32)
    logits = tf.reduce_sum(logits * target, axis=1)  # 只保留取样位置的值
    logits = tf.reshape(logits, [-1, 1])

    logits = tf.clip_by_value(tf.reshape(tf.sigmoid(logits), [-1, 1]), 1e-6, 0.999999)  # (bc*seq_len,1)

    return logits, attention_list

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

def frame2shot(vid,segment_info,scores):
    # 输入N*vlength的帧得分，以及对应视频的分段情况，输出同样形状的keyshot_labels
    # keyshot_labels将所有被选入summary的帧标记为1，其他标记为0
    cps = np.array(segment_info[vid])
    keyshot_labels = []
    for i in range(len(scores)):
        y = scores[i]
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        lists = [(y[cps[idx]:cps[idx + 1]], cps[idx]) for idx in range(len(cps) - 1)]
        segments = [tuple([np.average(i[0]), len(i[0]), i[1]]) for i in lists]
        value, weight, start = zip(*segments)
        max_weight = int(0.15 * len(y))
        chosen = knapSack(max_weight, weight, value, len(weight))
        keyshots = np.zeros(len(y))
        chosen = [int(j) for i, j in enumerate(chosen)]
        for i, j in enumerate(chosen):
            if (j == 1):
                keyshots[start[int(i)]:start[int(i)] + weight[int(i)]] = 1
        keyshot_labels.append(keyshots)
    keyshot_labels = np.array(keyshot_labels).squeeze()
    return keyshot_labels

def evaluation_frame(pred_scores, test_vids, segment_info, score_record):
    # 基于帧水平的评估方式，对于一个预测和一个GT的帧得分序列，首先转换成分段的形式，然后选出keyshot，再转换回帧标签的形式
    # 计算每一组预测和GT的F1，求平均作为这一视频的F1

    # 首先将模型输出裁剪为对每个视频中的帧得分预测
    preds_c = list(pred_scores[0])
    for i in range(1, len(pred_scores)):
        preds_c = preds_c + list(pred_scores[i])
    y_preds = {}

    # 计算F1，每个视频都计算20个F1，求总平均值
    pos = 0
    PRE_values = []
    REC_values = []
    F1_values = []
    for vid in test_vids:
        vlength = len(score_record[vid]['labels'])
        y_pred = np.array(preds_c[pos:pos+vlength])
        y_pred = np.expand_dims(y_pred,0)
        y_preds[vid] = y_pred
        pos += vlength
        label_pred = frame2shot(vid, segment_info, y_preds[vid])
        # label_trues = score_record[vid]['keyshot_labels']
        label_trues = [frame2shot(vid,segment_info,np.array(score_record[vid]['scores_avg']).reshape((1,vlength)))]
        for i in range(len(label_trues)):
            precision = np.sum(label_pred * label_trues[i]) / (np.sum(label_pred) + 1e-6)
            recall = np.sum(label_pred * label_trues[i]) / (np.sum(label_trues[i]) + 1e-6)
            PRE_values.append(precision)
            REC_values.append(recall)
            F1_values.append(2 * precision * recall / (precision + recall + 1e-6))
    PRE_values = np.array(PRE_values)
    REC_values = np.array(REC_values)
    F1_values = np.array(F1_values)

    return np.mean(PRE_values), np.mean(REC_values), np.mean(F1_values)

def model_search(model_save_dir, kfold=False):
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
    model_to_restore.sort(key=takestep)

    if kfold:
        # 只取最高F1的模型
        model_kfold = []
        f1s = []
        for name in model_to_restore:
            f1 = name.split('-')[-1]
            if f1.startswith('F'):
                f1s.append(float(f1.split('F')[-1]))
        f1_max = np.array(f1s).max()
        for name in model_to_restore:
            f1 = name.split('-')[-1]
            if f1.startswith('F') and float(f1.split('F')[-1]) >= f1_max:
                model_kfold.append(name)
        model_to_restore = model_kfold

    return model_to_restore

def run_training(data_train, data_test, segment_info, score_record, model_path, test_mode):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        features_holder = tf.placeholder(tf.float32, shape=(BATCH_SIZE * GPU_NUM, SEQ_LEN, D_INPUT))
        scores_holder = tf.placeholder(tf.float32, shape=(BATCH_SIZE * GPU_NUM, SEQ_LEN))
        labels_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,))
        sample_poses_holder = tf.placeholder(tf.int32,shape=(BATCH_SIZE * GPU_NUM,))
        mask_holder = tf.placeholder(tf.float32,shape=(NUM_HEADS, BATCH_SIZE * GPU_NUM, SEQ_LEN))
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
        attention_list = []
        for gpu_index in range(GPU_NUM):
            with tf.device('/gpu:%d' % gpu_index):
                features = features_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                labels = labels_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,]
                scores_avg = scores_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                sample_poses = sample_poses_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,]
                multihead_mask = mask_holder[:, gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]  # 从bc维切割

                # predict scores
                logits, atlist_one = score_pred(features, scores_avg, sample_poses, multihead_mask, dropout_holder, training_holder)
                logits_list.append(logits)
                attention_list += atlist_one  # 逐个拼接各个卡上的attention_list
                # calculate loss & gradients
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                # loss = tower_loss_huber(loss_name_scope, logits, labels)
                loss = tower_loss(loss_name_scope,logits,labels)
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
            logging.info(' Ckpt Model Restoring: '+model_path)
            saver_overall.restore(sess, model_path)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        train_scheme = train_scheme_build_v3(data_train, SEQ_LEN)
        # epoch_step = math.ceil(len(train_scheme[0]) / (BATCH_SIZE * GPU_NUM / 2))  # 所有正样本都计算过一次作为一个epoch
        # 正负样本比例可变的版本
        epoch_step = math.ceil(len(train_scheme[0]) / (BATCH_SIZE * GPU_NUM * POS_RATIO)) # 所有正样本都计算过一次作为一个epoch

        test_scheme, test_vids = test_scheme_build(data_test,SEQ_LEN)
        max_test_step = math.ceil(len(test_scheme) / BATCH_SIZE / GPU_NUM)

        # Beging training
        ob_loss = []
        timepoint = time.time()
        for step in range(MAXSTEPS):
            features_b, scores_avg_b, labels_b, sample_poses_b, mask_b = get_batch_train_v2(data_train, train_scheme,
                                                                                 step,GPU_NUM,BATCH_SIZE,SEQ_LEN)
            observe = sess.run([train_op] + loss_list + logits_list + attention_list + [global_step, lr],
                               feed_dict={features_holder: features_b,
                                          scores_holder: scores_avg_b,
                                          labels_holder: labels_b,
                                          sample_poses_holder: sample_poses_b,
                                          mask_holder: mask_b,
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
                    features_b, scores_avg_b, labels_b, sample_poses_b, mask_b = get_batch_test(data_test, test_scheme,
                                                                                       test_step, GPU_NUM, BATCH_SIZE, SEQ_LEN)
                    logits_temp_list = sess.run(logits_list, feed_dict={features_holder: features_b,
                                                                        scores_holder: scores_avg_b,
                                                                        sample_poses_holder: sample_poses_b,
                                                                        mask_holder: mask_b,
                                                                        training_holder: False,
                                                                        dropout_holder: 0})
                    for preds in logits_temp_list:
                        pred_scores.append(preds.reshape((-1)))
                p, r, f = evaluation_frame(pred_scores, test_vids, segment_info, score_record)
                logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f))
                if test_mode == 1:
                    return f
    return 0

def main(self):
    # load data
    score_record, video_category, segment_info = load_info(SCORE_PATH, VCAT_PATH, SEGINFO_PATH)
    subsets = load_feature_5fold(score_record, video_category, FEATURE_DIR)

    # split data
    data_train = {}
    data_valid = {}
    data_test = {}
    for i in range(K_FOLD_MODE, K_FOLD_MODE + 3):
        j = i % 5
        data_train.update(subsets[j])
    data_valid.update(subsets[(K_FOLD_MODE + 3) % 5])
    data_test.update(subsets[(K_FOLD_MODE + 4) % 5])
    f1_train = max_f1_estimate(score_record, list(data_train.keys()))
    f1_valid = max_f1_estimate(score_record, list(data_valid.keys()))
    f1_test = max_f1_estimate(score_record, list(data_test.keys()))
    logging.info('-' * 20 + 'K-fold: ' + str(K_FOLD_MODE) + '-' * 20)
    logging.info('Train Set Average Estimated F1: ' + str(f1_train.mean()))
    logging.info('Valid Set Average Estimated F1: ' + str(f1_valid.mean()))
    logging.info('Test Set Average Estimated F1: ' + str(f1_test.mean()))
    logging.info('-' * 50 + '\n')

    model_scores = {}
    for i in range(REPEAT_TIMES):
        model_save_dir = model_save_base + hp.msd + '_%d/' % i
        models_to_restore = model_search(model_save_dir, kfold=True)
        for i in range(len(models_to_restore)):
            logging.info('-' * 20+str(i)+': '+models_to_restore[i].split('/')[-1]+'-' * 20)
            ckpt_model_path = models_to_restore[i]
            f1 = run_training(data_train, data_test, segment_info, score_record, ckpt_model_path, 1)  # for training
            model_scores[ckpt_model_path] = f1

    f1_list = []
    for name in model_scores.keys():
        f1_list.append(model_scores[name])
    logging.info(str(f1_list))
    logging.info('Result: '+str(np.array(f1_list).mean()))

if __name__ == "__main__":
    tf.app.run()

