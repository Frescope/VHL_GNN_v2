import h5py
import json
import pickle
import numpy as np
import torch as t
import networkx as nx
import scipy.io
import os
from networkx.algorithms import bipartite
# from tools.cpd_auto import cpd_auto

def load_json(filename):
    with open(filename, encoding='utf8') as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def save_pickle(object,filename):
    with open(filename, 'wb') as f:
        pickle.dump(object,f)

config=load_json("/data/linkang/VHL_GNN/utc/config.json")
# for kind in ["C3D","resnet_avg"]:
#     print(kind)
#     for video_id in ["V1","V2","V3","V4"]:
#         f=h5py.File('/data/linkang/VHL_GNN/utc/features/'+video_id+'_'+kind+'.h5','r')
#         feature=f['feature'][()]
#         frame_num=feature.shape[0]
#         print(frame_num)
#
#         K=feature
#         K=np.dot(K,K.T)
#
#         cps,_=cpd_auto(K,config["max_segment_num"]-1,1,desc_rate=1,verbose=False,lmax=config["max_frame_num"]-1) #int(K.shape[0]/25)
#         seg_num=len(cps)+1
#
#         assert seg_num<=config["max_segment_num"]
#
#         seg_points=cps
#         seg_points=np.insert(seg_points,0,0)
#         seg_points=np.append(seg_points,frame_num)
#
#         segments=[]
#         for i in range(seg_num):
#             segments.append(np.arange(seg_points[i],seg_points[i+1],1,dtype=np.int32))
#
#         assert len(segments)<=config["max_segment_num"]
#
#         for seg in segments:
#             assert len(seg)<=config["max_frame_num"]
#
#         seg_len=np.zeros((config["max_segment_num"]),dtype=np.int32)
#         for index,seg in enumerate(segments):
#             seg_len[index]=len(seg)
#
#         # features
#
#         for seg in segments:
#             for frame in seg:
#                 assert frame<frame_num
#
#         features=t.zeros((config["max_segment_num"],config["max_frame_num"],4096 if kind=="C3D" else 2048))
#         for seg_index,seg in enumerate(segments):
#             for frame_index,frame in enumerate(seg):
#                 features[seg_index,frame_index]=t.tensor(feature[frame])
#                 # features[seg_index,frame_index]=F.avg_pool1d(t.tensor(feature[frame]).unsqueeze(0).unsqueeze(0),kernel_size=2,stride=2)
#
#
#         f=h5py.File('../data/processed/'+video_id+'_'+kind+'.h5','w')
#         f.create_dataset('features', data=features)
#         f.create_dataset('seg_len', data=seg_len)
#
#         f.close()

# kind = "resnet_avg"
# for video_id in ["V1","V2","V3","V4"]:
#     # test
#     f2=h5py.File('/data/linkang/VHL_GNN/utc/processed/'+video_id+'_'+kind+'.h5','r')
#     feature2 = f2['features'][()]
#     seg_len2 = f2['seg_len'][()]
#     print(np.array(seg_len2).sum())

def load_Tags(Tags_path):
    # 从Tags中加载每个视频中每个shot对应的concept标签
    Tags = []
    Tags_raw = scipy.io.loadmat(Tags_path)
    Tags_tmp1 = Tags_raw['Tags'][0]
    for i in range(4):
        Tags_tmp2 = Tags_tmp1[i][0]
        shot_labels = np.zeros((0, 48))
        for j in range(len(Tags_tmp2)):
            shot_label = Tags_tmp2[j][0][0].reshape((1, 48))
            shot_labels = np.vstack((shot_labels, shot_label))
        Tags.append(shot_labels)
        print(i, shot_labels.shape)
    return Tags

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

def consistency_analysis(Tags, base):
    # 基于F1-score验证user-summary与oracle-summary之间的一致性
    def concept(path):
        return path.split('/')[-1].split('_')[0:2]
    def f1_score(Tags,vid,l1,l2):
        sim_mat = similarity_compute(Tags, vid, l1, l2)
        weight = shot_matching(sim_mat)
        precision = weight / len(l1)
        recall = weight / len(l2)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    oracle_base = base + 'Query-Focused_Summaries/Oracle_Summaries/P0'
    user_base = base + 'Query-Focused_Summaries/User_Summaries/P0'
    U1_U2 = []
    U1_U3 = []
    U2_U3 = []
    U1_O = []
    U2_O = []
    U3_O = []
    for vid in range(1,5):
        # 整理顺序
        oracle_list = []
        user1_list = []
        user2_list = []
        user3_list = []
        for root, dirs, files in os.walk(oracle_base+str(vid)):
            for file in files:
                if file.endswith('_oracle.txt'):
                    oracle_list.append(os.path.join(root,file))
        for root, dirs, files in os.walk(user_base+str(vid)):
            for file in files:
                if file.endswith('_user1.txt'):
                    user1_list.append(os.path.join(root,file))
                if file.endswith('_user2.txt'):
                    user2_list.append(os.path.join(root, file))
                if file.endswith('_user3.txt'):
                    user3_list.append(os.path.join(root,file))
        oracle_list.sort()
        user1_list.sort()
        user2_list.sort()
        user3_list.sort()
        # 读取summary
        summary_num = len(oracle_list)
        for i in range(summary_num):
            oracle_sum = oracle_list[i]
            user1_sum = user1_list[i]
            user2_sum = user2_list[i]
            user3_sum = user3_list[i]
            # check
            assert concept(oracle_sum) == concept(user1_sum) == concept(user2_sum) == concept(user3_sum)
            # read
            with open(oracle_sum,'r') as f:
                label_o = []
                for line in f.readlines():
                    label_o.append(int(line.strip())-1)
            with open(user1_sum,'r') as f:
                label_1 = []
                for line in f.readlines():
                    label_1.append(int(line.strip())-1)
            with open(user2_sum,'r') as f:
                label_2 = []
                for line in f.readlines():
                    label_2.append(int(line.strip())-1)
            with open(user3_sum,'r') as f:
                label_3 = []
                for line in f.readlines():
                    label_3.append(int(line.strip())-1)
            # compute
            U1_U2.append(f1_score(Tags,vid,label_1,label_2))
            U1_U3.append(f1_score(Tags,vid,label_1,label_3))
            U2_U3.append(f1_score(Tags,vid,label_2,label_3))
            U1_O.append(f1_score(Tags,vid,label_1,label_o))
            U2_O.append(f1_score(Tags,vid,label_2,label_o))
            U3_O.append(f1_score(Tags,vid,label_3,label_o))
            print(vid,i)
    u12 = np.array(U1_U2).mean()
    u13 = np.array(U1_U3).mean()
    u23 = np.array(U2_U3).mean()
    u1o = np.array(U1_O).mean()
    u2o = np.array(U2_O).mean()
    u3o = np.array(U3_O).mean()
    print(u12,u13,u23)
    print(u1o,u2o,u3o)

Tags = load_Tags("/data/linkang/VHL_GNN/utc/Tags.mat")

# consistency_analysis(Tags, "/data/linkang/VHL_GNN/utc/origin_data/")

embedding = load_pickle("/data/linkang/VHL_GNN/utc/processed/query_dictionary.pkl")
print()

