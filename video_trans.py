# 实验9，模仿seq2seq方法将视频特征序列变为concept序列，使用encoder-decoder结构预测每个shot与各个concept的相关性
# 在encoder中将concept的嵌入表征与shot表征共同作为节点输入，将shot对应的表征作为decoder的输入
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
import networkx as nx

class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='3',type=str)
    parser.add_argument('--num_heads',default=8,type=int)
    parser.add_argument('--num_blocks',default=6,type=int)
    parser.add_argument('--seq_len',default=11,type=int)
    parser.add_argument('--bc',default=10,type=int)
    parser.add_argument('--dropout',default='0.1',type=float)
    parser.add_argument('--gpu_num',default=1,type=int)
    parser.add_argument('--msd', default='utc_SA', type=str)
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--lr_noam', default=1e-6, type=float)
    parser.add_argument('--warmup', default=4000, type=int)
    parser.add_argument('--maxstep', default=40000, type=int)
    parser.add_argument('--pos_ratio',default=0.5, type=float)
    parser.add_argument('--multimask',default=0, type=int)
    parser.add_argument('--kfold',default=1,type=int)
    parser.add_argument('--repeat',default=1,type=int)
    parser.add_argument('--eval_epoch',default=3,type=int)

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





if hp.server == 0:
    # path for JD server
    FEATURE_BASE = r'/public/data1/users/hulinkang/utc/features/'
    LABEL_BASE = r'/public/data1/users/hulinkang/utc/origin_data/Global_Summaries/'
    QUERY_SUM_BASE = r'/public/data1/users/hulinkang/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    EMBEDDING_PATH = r'/public/data1/users/hulinkang/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/public/data1/users/hulinkang/utc/concept_embeddding/'
    TAGS_PATH = r'/public/data1/users/hulinkang/utc/Tags.mat'
    model_save_base = r'/public/data1/users/hulinkang/model_HL_utc_query/'
    ckpt_model_path = r'/public/data1/users/hulinkang/model_HL_v4/utc_SA/'
else:
    # path for USTC servers
    FEATURE_BASE = r'/data/linkang/VHL_GNN/utc/features/'
    LABEL_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Global_Summaries/'
    QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    EMBEDDING_PATH = r'/data/linkang/VHL_GNN/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/data/linkang/VHL_GNN/utc/concept_embeddding/'
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
        # feature_path = feature_base + 'V%d_C3D.h5' % vid
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

Tags = load_Tags(TAGS_PATH)
data = load_feature_4fold(FEATURE_BASE, LABEL_BASE, Tags)