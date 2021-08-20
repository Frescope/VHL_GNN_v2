# trans3预测参数测试
import os
import json
import math
import argparse
import scipy.io
import numpy as np
import logging
import pickle
import networkx as nx

class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default=1, type=int)
    parser.add_argument('--msd', default='trans3_6b_8h_25s_10l_10d_l1144_p2575', type=str)

    parser.add_argument('--pred_ratio_lo', default=0.00, type=float)
    parser.add_argument('--pred_ratio_hi', default=1.00, type=float)
    parser.add_argument('--pred_ratio_step', default=0.05, type=float)

    parser.add_argument('--test_mode', default=1, type=int)

hparams = Path()
parser = hparams.parser
hp = parser.parse_args()

if hp.server == 0:
    # path for JD server
    TAGS_PATH = r'/public/data1/users/hulinkang/utc/Tags.mat'
    QUERY_SUM_BASE = r'/public/data1/users/hulinkang/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/public/data1/users/hulinkang/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'/public/data1/users/hulinkang/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/public/data1/users/hulinkang/utc/concept_embeddding/'
    MODEL_SAVE_BASE = r'/public/data1/users/hulinkang/model_HL_utc_query/'
elif hp.server == 1:
    # path for USTC servers
    TAGS_PATH = r'/data/linkang/VHL_GNN/utc/Tags.mat'
    QUERY_SUM_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'/data/linkang/VHL_GNN/utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'/data/linkang/VHL_GNN/utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'/data/linkang/VHL_GNN/utc/concept_embeddding/'
    MODEL_SAVE_BASE = r'/data/linkang/model_HL_v4/'
else:
    # path for JD A100 Server
    TAGS_PATH = r'../utc/Tags.mat'
    QUERY_SUM_BASE = r'../utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'
    CONCEPT_DICT_PATH = r'../utc/origin_data/Dense_per_shot_tags/Dictionary.txt'
    CONCEPT_TXT_EMB_PATH = r'../utc/processed/query_dictionary.pkl'
    CONCEPT_IMG_EMB_DIR = r'../utc/concept_embeddding/'
    MODEL_SAVE_BASE = r'/home/models/'

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

def MM_norm(preds):
    # 1D min-max normalization
    return (preds - preds.min()) / (preds.max() - preds.min())

def evaluation(outputs, Tags, query_summary, concepts, pred_ratio, appendix_list):
    model_scores = {}
    for vid in outputs:
        f1_scores = []
        # logging.info('Video: ' + vid)
        for count in outputs[vid]:
            # logging.info('Model_count: ' + count)
            c_predictions = np.array(outputs[vid][count]['c_predictions'])
            s_predictions = np.array(outputs[vid][count]['s_predictions'])
            p_predictions = np.array(outputs[vid][count]['p_predictions'])
            vlength = len(c_predictions)
            summary = query_summary[vid]
            hl_num = math.ceil(vlength * 0.02)

            # front-preferrence
            seg_len = math.ceil(vlength / len(appendix_list))
            appendix = []
            for num in appendix_list:
                appendix += [num] * seg_len
            appendix = np.array(appendix[:vlength])

            PRE_values = []
            REC_values = []
            F1_values = []
            for query in summary:
                shots_gt = summary[query]
                # q_ind = queries[vid].index(query)
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
                scores = (pred_c1 + pred_c2) / 2 * pred_ratio + \
                         candidate * (1 - pred_ratio)
                scores += appendix

                scores_indexes = scores.reshape((-1, 1))
                scores_indexes = np.hstack((scores_indexes, np.array(range(len(scores))).reshape((-1, 1))))
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
            PRE_value = np.array(PRE_values).mean()
            REC_value = np.array(REC_values).mean()
            F1_value = np.array(F1_values).mean()
            # logging.info('Precision: %.3f, Recall: %.3f, F1: %.3f' % (PRE_value, REC_value, F1_value))
            f1_scores.append(F1_value)
        model_scores[vid] = f1_scores

    scores_all = 0
    for vid in model_scores:
        f1_scores = model_scores[vid]
        logging.info('Vid: %s, Mean: %.3f, Scores: %s' %
                     (vid, np.array(f1_scores).mean(), str(f1_scores)))
        scores_all += np.array(f1_scores).mean()
    logging.info('Ratio: %.2f, Overall Results: %.3f' % (pred_ratio, scores_all / 4))
    logging.info(str(appendix_list))
    return scores_all / 4

def main():
    Tags = load_Tags(TAGS_PATH)
    queries, query_summary = load_query_summary(QUERY_SUM_BASE)
    concepts, concept_embedding = load_concept(CONCEPT_DICT_PATH, CONCEPT_TXT_EMB_PATH, CONCEPT_IMG_EMB_DIR)

    with open(MODEL_SAVE_BASE + hp.msd + '_test_outputs.json', 'r') as file:
        outputs = json.load(file)

    # Test prediction balance ratio
    if hp.test_mode == 0:
        f1_max = 0
        ratio_max = 0
        pred_ratio = hp.pred_ratio_lo
        while pred_ratio <= hp.pred_ratio_hi:
            f1_mean = evaluation(outputs, Tags, query_summary, concepts, pred_ratio, 0, 0)
            if f1_max < f1_mean:
                f1_max = f1_mean
                ratio_max = pred_ratio
            pred_ratio += hp.pred_ratio_step
        logging.info('\nRatio: %.2f, Max F1: %.3f' % (ratio_max, f1_max))

    # Test appendix
    else:
        test_list = [[0.1, 0],
                     [0.2, 0],
                     [0.3, 0],
                     [0.4, 0],
                     [0.5, 0],
                     [0.2, 0.1, 0],
                     [0.3, 0.15, 0],
                     [0.4, 0.2, 0],
                     [0.4,0.3,0.2,0.1,0],
                     [0.5,0.4,0.3,0.2,0.1,0],
                     [0.6,0.5,0.4,0.3,0.2,0.1,0],
                     [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]]

        f1_max = 0
        ind_max = 0
        for i in range(len(test_list)):
            f1_mean = evaluation(outputs, Tags, query_summary, concepts, 0.75, test_list[i])
            if f1_max < f1_mean:
                f1_max = f1_mean
                ind_max = i
        logging.info('\nMax F1: %.3f' % f1_max)
        logging.info(str(test_list[ind_max]))

if __name__ == '__main__':
    main()