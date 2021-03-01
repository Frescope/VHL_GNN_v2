# 在feat_preproc之后执行，从feat_preproc中获取初步生成的score_record，选择出关键帧以及关键片段，重新保存

import os
import numpy as np
import math
import json
from tools.knapsack_iter import knapSack
import copy

SCORE_PATH = r'/data/linkang/VHL_GNN/tvsum_score_record.json'
SEGINFO_PATH = r'/data/linkang/VHL_GNN/tvsum_segment_info.json'
KEY_FRAME_RATIO = 0.35 # 整体上大约有33%的帧标记为keyframe

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def frame2shot(vid,segment_info,scores):
    # 输入N*vlength的帧得分，以及对应视频的分段情况，输出同样形状的keyshot_labels
    # keyshot_labels将所有被选入summary的帧标记为1，其他标记为0
    cps = np.array(segment_info[vid])
    keyshot_labels = []
    for i in range(len(scores)):
        y = scores[i]
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

def label_process(score_record, segment_info, keyframe_ratio):
    # 1. 从scores_avg中根据一定的比例选择出keyframe
    # 2. 根据分段信息与标注，生成每个标注对应的summary，以帧的二分类标签形式呈现
    # 3. 对于每个视频，使用贪心算法找出一组能够使平均F1最大化的帧标注，作为训练GT

    # select keyframe
    vids = list(score_record.keys())
    key_ratios = []
    for vid in vids:
        scores_avg = np.array(score_record[vid]['scores_avg'])
        score_list = list(scores_avg)
        score_list.sort(reverse=True)
        threshold = score_list[math.ceil(len(scores_avg) * keyframe_ratio)]  # 得分大于门限的帧设为keyframe
        labels = (scores_avg > threshold).astype(int)
        score_record[vid]['labels'] = labels
        key_ratios.append(np.mean(labels))
    key_ratios = np.array(key_ratios)
    # print(keyframe_ratio,np.min(key_ratios),np.max(key_ratios),np.mean(key_ratios))

    # convert frame score to keyshot
    for vid in vids:
        scores = np.array(score_record[vid]['scores'])  # 20*vlength
        keyshot_labels = frame2shot(vid,segment_info,scores)  # 20*vlength
        score_record[vid]['keyshot_labels'] = keyshot_labels

    # select label_greedy
    for vid in vids:
        savg = score_record[vid]['scores_avg']
        keyshot_labels = score_record[vid]['keyshot_labels']
        label_greedy, f1 = greedy_label_build(savg, keyshot_labels)
        score_record[vid]['label_greedy'] = label_greedy
        print(vid,len(label_greedy),np.sum(label_greedy),"%.4f"%(np.sum(label_greedy) / len(label_greedy)), f1)

    return score_record

def f1_calc(pred,gts):
    # 计算pred与所有gt的平均f1
    f1s = []
    for gt in gts:
        precision = np.sum(pred * gt) / (np.sum(pred) + 1e-6)
        recall = np.sum(pred * gt) / (np.sum(gt) + 1e-6)
        f1s.append(2 * precision * recall / (precision + recall + 1e-6))
    return np.array(f1s).mean()

def greedy_label_build(savg, keyshot_lables):
    # 寻找一个可以使得与multiple annotations的F1均值最大化的标签序列
    # 输入scores_avg作为搜索序列

    def takescore(elem):
        return elem[1]

    F1 = 0
    label= np.zeros(len(savg))  # keyframe集合
    frame_score = list(enumerate(savg))
    frame_score.sort(key=takescore, reverse=True)  # 将（index，score）的序列排序
    while(True):
        F1_temp = F1
        ind_temp = 0
        for i in range(len(frame_score)):
            f_ind, f_sc = frame_score[i]
            label_new = copy.deepcopy(label)
            label_new[f_ind] = 1
            F1_new = f1_calc(label_new, keyshot_lables)
            if F1_new > F1_temp:
                F1_temp = F1_new
                ind_temp = i
            del label_new
        if F1_temp > F1:
            index, _ = frame_score.pop(ind_temp)
            F1 = F1_temp
            label[index] = 1
        else:
            break  # 新增任一帧都无法继续增大F1，则结束
    return label, F1

def max_f1_estimate(score_record, segment_info):
    # 使用scores_avg作为预测，计算与各个summary的F1，作为对模型可能达到的最大F1的估计
    f1_overall = []
    f1_overall_greedy = []
    vids = list(score_record.keys())
    for vid in vids:
        vlength = len(score_record[vid]['labels'])
        savg = np.array(score_record[vid]['scores_avg']).reshape((1,vlength))
        label_savg = frame2shot(vid,segment_info,savg)
        label_trues = score_record[vid]['keyshot_labels']
        pres = []
        recs = []
        f1s = []
        for i in range(len(label_trues)):
            precision = np.sum(label_savg * label_trues[i]) / (np.sum(label_savg) + 1e-6)
            recall = np.sum(label_savg * label_trues[i]) / (np.sum(label_trues[i]) + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            pres.append(precision)
            recs.append(recall)
            f1s.append(f1)
        pre = np.array(pres).mean()
        rec = np.array(recs).mean()
        f1 = np.array(f1s).mean()

        # check label_greedy
        label_greedy = np.array(score_record[vid]['label_greedy'])
        f1_greedy = f1_calc(label_greedy,label_trues)
        print(vid, pre, rec, f1, f1_greedy)
        f1_overall.append(f1)
        f1_overall_greedy.append(f1_greedy)
    print('F1: ',np.array(f1_overall).mean(), np.array(f1_overall_greedy).mean())

if __name__ == '__main__':
    with open(SCORE_PATH,'r') as file:
        score_record = json.load(file)
    with open(SEGINFO_PATH, 'r') as file:
        segment_info = json.load(file)

    # score_record_new = label_process(score_record, segment_info, KEY_FRAME_RATIO)
    #
    # with open(SCORE_PATH, 'w') as file:
    #     json.dump(score_record_new, file, cls=NpEncoder)

    score_record_new = score_record
    max_f1_estimate(score_record_new, segment_info)

    print('Done !')