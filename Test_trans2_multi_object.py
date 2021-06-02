# video_trans_v2的多目标学习测试，引入每个shot对应的concept标签，训练模型做多标签分类，将loss加入总loss中去

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
from Test_trans2_multi_object_transformer import transformer
import networkx as nx

class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1',type=str)
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=6,type=int)
    parser.add_argument('--seq_len',default=20,type=int)
    parser.add_argument('--bc',default=20,type=int)
    parser.add_argument('--dropout',default='0.1',type=float)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--msd', default='video_trans', type=str)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--lr_noam', default=1e-6, type=float)
    parser.add_argument('--warmup', default=1500, type=int)
    parser.add_argument('--maxstep', default=10000, type=int)
    parser.add_argument('--pos_ratio', default=0.1, type=float)
    parser.add_argument('--concept_ratio', default=0.25, type=float)
    parser.add_argument('--multimask',default=0, type=int)
    parser.add_argument('--repeat',default=3,type=int)
    parser.add_argument('--eval_epoch',default=1,type=int)

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
D_OUTPUT = 48  # label_S1对应48，label_S2对应45
CONCEPT_NUM = 48
MAX_F1 = 0.2
GRAD_THRESHOLD = 10.0  # gradient threshold

LOAD_CKPT_MODEL = False
MIN_TRAIN_STEPS = 0
PRESTEPS = 0

if hp.server == 0:
    # path for JD server
    FEATURE_BASE = r'/public/data1/users/hulinkang/utc/features/'
    TAGS_PATH = r'/public/data1/users/hulinkang/utc/Tags.mat'
    LABEL_PATH = r'/public/data1/users/hulinkang/utc/videotrans_label_s1.json'
    CONCEPT_LABEL_PATH = r'/public/data1/users/hulinkang/utc/concept_label.json'
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
    LABEL_PATH = r'/data/linkang/VHL_GNN/utc/videotrans_label_s1.json'
    CONCEPT_LABEL_PATH = r'/data/linkang/VHL_GNN/utc/concept_label.json'
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

def load_feature_4fold(feature_base, labe_path, concept_label_path, Tags):
    # 注意label对应的concept是按照字典序排列的
    with open(labe_path, 'r') as file:
        labels = json.load(file)
    with open(concept_label_path, 'r') as file:
        concept_labels = json.load(file)

    data = {}
    for vid in range(1,5):
        data[str(vid)] = {}
        vlength = len(Tags[vid-1])
        # feature

        # feature_path = feature_base + 'V%d_resnet_avg.h5' % vid
        # feature_path = feature_base + 'V%d_C3D.h5' % vid
        # f = h5py.File(feature_path, 'r')
        # feature = f['feature'][()][:vlength]

        feature_path = feature_base + 'V%d_I3D.npy' % vid
        feature = np.load(feature_path)

        data[str(vid)]['feature'] = feature

        # label
        label = np.array(labels[str(vid)])[:,:vlength].T
        # for s1
        # label = (label - label.min(0)) / (label.max(0) - label.min(0) + 1e-6)  # 归一化
        data[str(vid)]['label'] = label

        # concept label
        concept_label = np.array(concept_labels[str(vid)])
        data[str(vid)]['concept_label'] = concept_label

        logging.info('Vid: '+str(vid)+' Feature: '+str(feature.shape)+' Label: '+str(label.shape))
    return data

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

def train_scheme_build(data_train, seq_len):
    # 对于每个视频，标记出每个concept对应的正例和负例，分别为每个concept建立正例与负例集合，每个序列中只在一个concept对应的正例与负例集合中选择，目的是保证每个输入序列只针对一个concept进行训练
    # 每次按照一定比例从正例集合中取若干clip（按顺序），所有正例全部训练一次作为一个epoch，每个epoch后随机化正例与负例集合的顺序
    # 不同视频、不同concept的序列混合排列，但是必须保证每个序列中只有来自同一个视频的clip
    info_dict = {}
    for vid in data_train:
        label = data_train[vid]['label']
        info_dict[vid] = {}
        for cid in range(CONCEPT_NUM):
            label_concept = label[:, cid]
            concept_pos_list = list(np.where(label_concept > 0)[0])
            concept_neg_list = list(np.where(label_concept == 0)[0])
            random.shuffle(concept_pos_list)
            random.shuffle(concept_neg_list)
            info_dict[vid][str(cid)] = [concept_pos_list, concept_neg_list]

    # 按照固定比例选取正例与负例，取完所有正例为止，每个epoch更新一次训练列表
    train_scheme = []
    pos_num = math.ceil(seq_len * hp.pos_ratio)
    for vid in info_dict:
        for cid in range(CONCEPT_NUM):
            concept_pos_list, concept_neg_list = info_dict[vid][str(vid)]
            pos_ind = 0  # pos_list中的位置
            neg_ind = 0
            while(pos_ind < len(concept_pos_list)):
                clip_list = concept_pos_list[pos_ind : pos_ind + pos_num]
                clip_list += concept_neg_list[neg_ind : neg_ind + seq_len - len(clip_list)]  # 正例不足时用负例补足
                clip_list += concept_pos_list[0 : seq_len - len(clip_list)]  # 负例不足时做padding，一般不起作用
                clip_list.sort()

                # # test
                # label_temp = np.sum(data_train[vid]['label'], axis=1)[clip_list]
                # pl = np.where(label_temp > 0)[0]
                # nl = np.where(label_temp == 0)[0]
                # print(vid,pos_ind, 'pos: ',len(pl), pos_ind, 'neg: ',len(nl), neg_ind)

                pos_ind += pos_num
                neg_ind = (neg_ind + seq_len - pos_num) % len(concept_neg_list)  # 循环取负例
                train_scheme.append((vid, cid, clip_list))
    random.shuffle(train_scheme)
    return train_scheme

def get_batch_train(data_train, train_scheme, step, gpu_num, bc, seq_len):
    # 从train_scheme中获取gpu_num*bc个序列，每个长度seq_len，并返回每个clip的全局位置
    batch_num = gpu_num * bc
    features = []
    labels = []
    concept_labels = []
    positions = []
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(train_scheme)
        vid, cid, clip_list = train_scheme[pos]
        features.append(data_train[vid]['feature'][clip_list])
        labels.append(data_train[vid]['label'][clip_list])
        concept_labels.append(data_train[vid]['concept_label'][clip_list])
        positions.append(clip_list)
    features = np.array(features)
    labels = np.array(labels)
    concept_labels = np.array(concept_labels)
    positions = np.array(positions)
    scores = np.ones((batch_num, seq_len))
    return features, labels, concept_labels, positions, scores

def test_scheme_build(data_test, seq_len):
    # 依次输入测试集中所有clip，不足seqlen的要补足，在getbatch中补足不够一个batch的部分
    # (vid, seq_start, seq_end)形式
    test_scheme = []
    test_vids = []
    for vid in data_test:
        vlength = len(data_test[str(vid)]['label'])
        seq_num = math.ceil(vlength / seq_len)
        for i in range(seq_num):
            test_scheme.append((vid, i * seq_len, min(vlength,(i+1) * seq_len)))
        test_vids.append((vid, vlength))
    return test_scheme, test_vids

def get_batch_test(data_test, test_scheme, step, gpu_num, bc, seq_len):
    # 标记每个序列中的有效长度，并对不足一个batch的部分做padding
    # 不需要对序列水平上的padding做标记
    features = []
    labels = []
    concept_labels = []
    positions = []
    scores = []
    batch_num = gpu_num * bc
    for i in range(batch_num):
        pos = (step * batch_num + i) % len(test_scheme)
        vid, seq_start, seq_end = test_scheme[pos]
        vlength = len(data_test[str(vid)]['label'])
        padding_len = seq_len - (seq_end - seq_start)
        feature = data_test[str(vid)]['feature'][seq_start:seq_end]
        label = data_test[str(vid)]['label'][seq_start:seq_end]
        concept_label = data_test[str(vid)]['concept_label'][seq_start:seq_end]
        position = np.array(list(range(seq_start, seq_end)))
        score = np.ones(len(label))
        if padding_len > 0:
            feature_pad = np.zeros((padding_len, D_FEATURE))
            label_pad = np.zeros((padding_len, D_OUTPUT))
            concept_label_pad = np.zeros((padding_len, CONCEPT_NUM))
            position_pad = np.array([vlength] * padding_len)
            score_pad = np.zeros(padding_len)
            feature = np.vstack((feature, feature_pad))
            label = np.vstack((label, label_pad))
            concept_label = np.vstack((concept_label, concept_label_pad))
            position = np.hstack((position, position_pad))
            score = np.hstack((score, score_pad))
        features.append(feature)
        labels.append(label)
        concept_labels.append(concept_label)
        positions.append(position)
        scores.append(score)
    features = np.array(features)
    labels = np.array(labels)
    concept_labels = np.array(concept_labels)
    positions = np.array(positions)
    scores = np.array(scores)
    return features, labels, concept_labels, positions, scores

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

def tower_loss_multi_object(logits, labels, concept_logits, concete_labels):
    # logits & labels: bc*seq_len*48
    # concept_logits & concept_labels: bc*seq_len*48
    # 对每个concept，计算序列中所有clip的NCE-loss，对所有concept、所有序列求均值
    # 对每个shot，计算其logits与实际concept_label的交叉熵，求所有shot的均值
    # 合并上述两种loss

    logits = tf.transpose(logits, perm=(0,2,1))  # bc*48*seq_len
    logits = tf.reshape(logits, shape=(-1, hp.seq_len))  # (bc*48)*seq_len
    labels = tf.transpose(labels, perm=(0,2,1))
    labels = tf.reshape(labels, shape=(-1, hp.seq_len))
    labels_binary = tf.cast(tf.cast(labels, dtype=tf.bool), dtype=tf.float32)  # 转化为0-1形式，浮点数
    nce_pos = tf.reduce_sum(tf.exp(labels_binary * logits), axis=1)  # 分子
    nce_pos -= tf.reduce_sum((1 - labels_binary), axis=1)  # 减去负例（为零）取e后的值（为1）
    nce_all = tf.reduce_sum(tf.exp(logits), axis=1)   # 分母
    nce_loss = -tf.log((nce_pos / nce_all) + 1e-5)
    loss = tf.reduce_mean(nce_loss)

    y = concete_labels
    x = tf.clip_by_value(tf.sigmoid(concept_logits), 1e-6, 0.999999)
    concept_loss = -y * (tf.log(x)) - (1 - y) * tf.log(1 - x)
    concept_loss = tf.reduce_mean(concept_loss)

    w = hp.concept_ratio
    loss = (1 - w) * loss + w * concept_loss
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

def evaluation(pred_scores, queries, query_summary, Tags, test_vids, concepts):
    # 从每个视频的预测结果中，根据query中包含的concept选出相关程度最高的一组shot，匹配后计算f1，求所有query的平均结果
    preds_c = pred_scores[0]
    for i in range(1, len(pred_scores)):
        preds_c = np.vstack((preds_c, pred_scores[i]))
    pos = 0
    PRE_values = []
    REC_values = []
    F1_values = []
    for i in range(len(test_vids)):
        vid, vlength = test_vids[i]
        summary = query_summary[str(vid)]
        hl_num = math.ceil(vlength * 0.02)
        predictions = preds_c[pos : pos + vlength]
        pos += vlength
        for query in summary:
            shots_gt = summary[query]
            c1, c2 = query.split('_')

            # for s1
            ind1 = concepts.index(c1)
            ind2 = concepts.index(c2)
            scores = (predictions[:,ind1] + predictions[:,ind2]).reshape((-1))
            # # for s2
            # index = queries[str(vid)].index([c1, c2])
            # scores = predictions[:, index].reshape((-1))

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

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def model_clear(model_save_dir, max_f1):
    # 清除之前所有F1较小的模型
    models = []
    for name in os.listdir(model_save_dir):
        if name.endswith('.meta'):
            models.append(name.split('.meta')[0])
    for model in models:
        f1 = model.split('-')[-1]
        if f1.startswith('F') and float(f1.split('F')[-1]) < max_f1:
            file_path = os.path.join(model_save_dir, model) + '*'
            os.system('rm -rf %s' % file_path)
    return

def run_training(data_train, data_test, queries, query_summary, Tags, concepts, concept_embeeding, model_save_dir, test_mode):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    max_f1 = MAX_F1

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        features_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_FEATURE))
        labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, D_OUTPUT))
        concept_labels_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len, CONCEPT_NUM))
        positions_holder = tf.placeholder(tf.int32, shape=(hp.bc * hp.gpu_num, hp.seq_len))
        scores_src_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len + CONCEPT_NUM))
        scores_tgt_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, hp.seq_len))
        txt_emb_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, CONCEPT_NUM, D_TXT_EMB))
        img_emb_holder = tf.placeholder(tf.float32, shape=(hp.bc * hp.gpu_num, CONCEPT_NUM, D_IMG_EMB))
        dropout_holder = tf.placeholder(tf.float32, shape=())
        training_holder = tf.placeholder(tf.bool, shape=())

        # training operations
        lr = noam_scheme(hp.lr_noam, global_step, hp.warmup)
        opt_train = tf.train.AdamOptimizer(lr)

        # graph building
        tower_grads_train = []
        logits_list = []
        loss_list = []
        for gpu_index in range(hp.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                features = features_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                labels = labels_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                concept_labels = concept_labels_holder[gpu_index * hp.bc: (gpu_index + 1) * hp.bc]
                positions = positions_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                scores_src = scores_src_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                scores_tgt = scores_tgt_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                txt_emb = txt_emb_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]
                img_emb = img_emb_holder[gpu_index * hp.bc : (gpu_index+1) * hp.bc]

                # predict concept distribution
                logits, concept_logits = transformer(features, positions, scores_src, scores_tgt, txt_emb, img_emb,
                                     dropout_holder, training_holder, hp, CONCEPT_NUM)  # 输入的shot在所有concept上的相关性分布
                logits_list.append(logits)

                loss = tower_loss_multi_object(logits, labels, concept_logits, concept_labels)
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
        if LOAD_CKPT_MODEL:
            logging.info(' Ckpt Model Restoring: ' + CKPT_MODEL_PATH)
            saver_overall.restore(sess, CKPT_MODEL_PATH)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        train_scheme = train_scheme_build(data_train, hp.seq_len)
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
        timepoint = time.time()
        for step in range(hp.maxstep):
            features_b, labels_b, concept_labels_b, positions_b, scores_b = get_batch_train(data_train, train_scheme, step, hp.gpu_num, hp.bc, hp.seq_len)
            scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, CONCEPT_NUM))))  # encoder中开放所有concept节点
            scores_tgt_b = scores_b
            observe = sess.run([train_op] + loss_list + logits_list,
                               feed_dict={features_holder: features_b,
                                          labels_holder: labels_b,
                                          concept_labels_holder: concept_labels_b,
                                          positions_holder: positions_b,
                                          scores_src_holder: scores_src_b,
                                          scores_tgt_holder: scores_tgt_b,
                                          txt_emb_holder: txt_emb_b,
                                          img_emb_holder: img_emb_b,
                                          dropout_holder: hp.dropout,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1 + hp.gpu_num])
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == hp.maxstep:
                if step == 0 and test_mode == 0:
                    continue
                train_scheme = train_scheme_build(data_train, hp.seq_len)  # shuffle train scheme
                duration = time.time() - timepoint
                timepoint = time.time()
                loss_array = np.array(ob_loss)
                ob_loss.clear()
                logging.info(' Step %d: %.3f sec' % (step, duration))
                logging.info(' Evaluate: ' + str(step) + ' Epoch: ' + str(epoch))
                logging.info(' Average Loss: ' + str(np.mean(loss_array)) + ' Min Loss: ' + str(
                    np.min(loss_array)) + ' Max Loss: ' + str(np.max(loss_array)))
                if not int(epoch) % hp.eval_epoch == 0:
                    continue  # 增大测试间隔
                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                pred_scores = []  # 每个batch输出的预测得分
                for test_step in range(max_test_step):
                    features_b, labels_b, concept_labels_b, positions_b, scores_b = get_batch_test(data_test, test_scheme, test_step, hp.gpu_num, hp.bc, hp.seq_len)
                    scores_src_b = np.hstack((scores_b, np.ones((hp.gpu_num * hp.bc, CONCEPT_NUM))))  # encoder中开放所有concept节点
                    scores_tgt_b = scores_b
                    logits_temp_list = sess.run(logits_list, feed_dict={features_holder: features_b,
                                                                        labels_holder: labels_b,
                                                                        concept_labels_holder: concept_labels_b,
                                                                        positions_holder: positions_b,
                                                                        scores_src_holder: scores_src_b,
                                                                        scores_tgt_holder: scores_tgt_b,
                                                                        txt_emb_holder: txt_emb_b,
                                                                        img_emb_holder: img_emb_b,
                                                                        dropout_holder: hp.dropout,
                                                                        training_holder: False})
                    for preds in logits_temp_list:
                        pred_scores.append(preds.reshape((-1, D_OUTPUT)))
                p, r, f = evaluation(pred_scores, queries, query_summary, Tags, test_vids, concepts)
                logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f))

                if test_mode == 1:
                    return
                # save model
                if step > MIN_TRAIN_STEPS - PRESTEPS and f >= max_f1:
                    max_f1 = f
                    model_clear(model_save_dir, max_f1)
                    model_path = model_save_dir + 'S%d-E%d-L%.6f-F%.3f' % (step, epoch, np.mean(loss_array), f)
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

def main(self):
    # load data
    Tags = load_Tags(TAGS_PATH)
    data = load_feature_4fold(FEATURE_BASE, LABEL_PATH, CONCEPT_LABEL_PATH, Tags)
    queries, query_summary = load_query_summary(QUERY_SUM_BASE)
    concepts, concept_embedding = load_concept(CONCEPT_DICT_PATH, CONCEPT_TXT_EMB_PATH, CONCEPT_IMG_EMB_DIR)

    # evaluate all videos in turn
    for kfold in range(4):
        if kfold < 3:
            continue
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
        logging.info('*' * 50)

        # repeat
        for i in range(hp.repeat):
            model_save_dir = MODEL_SAVE_BASE + hp.msd + '_%d_%d/' % (kfold, i)
            logging.info('*' * 10 + str(i) + ': ' + model_save_dir + '*' * 10)
            logging.info('*' * 60)
            run_training(data_train, data_valid, queries, query_summary, Tags, concepts, concept_embedding,
                         model_save_dir, 0)
            logging.info('*' * 60)
        logging.info('^' * 60 + '\n')

if __name__ == '__main__':
    tf.app.run()






