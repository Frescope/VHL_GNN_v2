# 测试随机序列的F1值
import json
import os
import numpy as np

SCORE_RECORD_PATH = r'/public/data1/users/hulinkang/tvsum/score_record.json'
SEGINFO_PATH = r'/public/data1/users/hulinkang/tvsum/VHL_GNN_v2/tvsum_segment_info.json'


def knapSack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]

    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]


    best = K[n][W]

    amount = np.zeros(n);
    a = best;
    j = n;
    Y = W;

    while a > 0:
       while K[j][Y] == a:
           j = j - 1;

       j = j + 1;
       amount[j-1] = 1;
       Y = Y - wt[j-1];
       j = j - 1;
       a = K[j][Y];

    return amount

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

def evaluation(pred_scores, score_record, segment_info):
    # 计算F1，每个视频都只与一个标签计算
    pos = 0
    PRE_values = []
    REC_values = []
    F1_values = []
    for vid in score_record:
        vlength = len(score_record[vid]['keyshot_label_uni'])
        y_pred = np.array(pred_scores[pos:pos + vlength])
        y_pred = np.expand_dims(y_pred, 0)
        label_pred = frame2shot(vid, segment_info, y_pred)
        label_true = score_record[vid]['keyshot_labels']
        for i in range(len(label_true)):
            label_one = np.array(label_true[i])
            precision = np.sum(label_pred * label_one) / (np.sum(label_pred) + 1e-6)
            recall = np.sum(label_pred * label_one) / (np.sum(label_one) + 1e-6)
            PRE_values.append(precision)
            REC_values.append(recall)
            F1_values.append(2 * precision * recall / (precision + recall + 1e-6))
    PRE_values = np.array(PRE_values)
    REC_values = np.array(REC_values)
    F1_values = np.array(F1_values)

    return np.mean(PRE_values), np.mean(REC_values), np.mean(F1_values)

if __name__ == '__main__':
    with open(SEGINFO_PATH, 'r') as file:
        segment_info = json.load(file)
    with open(SCORE_RECORD_PATH, 'r') as file:
        score_record = json.load(file)

    total_length = 0
    for vid in score_record:
        vlength = len(score_record[vid]['keyshot_label_uni'])
        total_length += vlength
    random_scores = np.random.random(total_length)
    p,r,f = evaluation(list(random_scores), score_record, segment_info)
    print(p,r,f)
