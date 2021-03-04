# summe特征与标签的前处理，包括帧的截取与标签的制作，后续只进行特征提取即可
# 读取GT，抽取groundtruth
# 截取帧，获取对应的标注形成score_record，根据KTS分段结果和段内平均分求keyshot_label
# 用取平均+排序、帧上的贪心策略和分段上的贪心策略分别求出一组训练GT

import os
import numpy as np
import cv2
import math
import scipy.io
import json
import copy
from tools.knapsack_iter import knapSack

VIDEO_BASE = r'/data/linkang/SumMe/videos/'
FRAME_BASE = r'/data/linkang/SumMe/frame_2fps/'
GT_DIR = r'/data/linkang/SumMe/GT/'
GT_PATH = r'/data/linkang/SumMe/groundtruth.json'
SCORE_PATH = r'/data/linkang/VHL_GNN/summe_score_record.json'
SEGINFO_PATH = r'/data/linkang/VHL_GNN/summe_segment_info.json'
FRAME_RATE = 2  # 2fps

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

def load_mat(gt_dir,gt_path):
    groundtruth = {}
    for root, dirs, files in os.walk(gt_dir):
        for name in files:
            gt_data = scipy.io.loadmat(os.path.join(root,name))
            vid = name.split('.mat')[0]
            groundtruth[vid] = {}
            groundtruth[vid]['all_userIDs'] = gt_data['all_userIDs']
            groundtruth[vid]['segments'] = gt_data['segments']
            groundtruth[vid]['nFrames'] = gt_data['nFrames']
            groundtruth[vid]['video_duration'] = gt_data['video_duration']
            groundtruth[vid]['FPS'] = gt_data['FPS']
            groundtruth[vid]['gt_score'] = gt_data['gt_score']
            groundtruth[vid]['user_score'] = gt_data['user_score']
    with open(gt_path, 'w') as file:
        json.dump(groundtruth, file, cls=NpEncoder)
    print('Done !')

def load_info(path):
    # any info in json format
    with open(path, 'r') as file:
        info = json.load(file)
    return info

def frame_capture(groundtruth,vid,video_base,frame_base):
    video_path = video_base + vid + '.mp4'
    frame_dir = frame_base + vid + r'/'
    if os.path.isdir(frame_dir):
        print('Directory already exists !')
        return
    else:
        os.makedirs(frame_dir)

    # get annotations
    user_score = np.array(groundtruth[vid]['user_score'])  # 关于这一id的所有标注
    user_score = user_score.T

    # capturing
    scores = np.zeros((len(user_score),0))
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_interval = round(fps / FRAME_RATE)  # 截取帧的间隔
    rval, frame = vc.read()
    frame_count = 0
    while rval:
        if frame_count % frame_interval == 0:
            frame_time = round(frame_count / fps * 10)  # 帧对应的秒数*10
            frame_path = frame_dir + str(frame_time).zfill(6) + '.jpg'
            cv2.imwrite(frame_path, frame)
            score_frame = user_score[:,frame_count:frame_count + 1]  # 这一帧对应的所有标注
            scores = np.hstack((scores,score_frame))
        if frame_count % 1000 == 0 and frame_count > 0:
            print('Frames: ',frame_count)
        rval, frame = vc.read()
        frame_count += 1
    vc.release()
    print('Frames & Scores Extracted: ',vid,frame_count,scores.shape)
    # feat = np.load(r'/data/linkang/VHL_GNN/tvsum_feature_googlenet_2fps/'+vid+r'_googlenet_2fps.npy')
    # print(feat.shape)
    return scores

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

def testing_gt_build(score_record, segment_info):
    vids = list(score_record.keys())
    print('Testing GT: ')
    for vid in vids:
        scores = np.array(score_record[vid]['scores'])  # N*vlength
        keyshot_labels = frame2shot(vid, segment_info, scores)  # 20*vlength
        score_record[vid]['keyshot_labels'] = keyshot_labels
        vlength = keyshot_labels.shape[-1]
        summary_num = np.sum(keyshot_labels, axis=1).mean()
        print(vid,keyshot_labels.shape,summary_num/vlength)
    return score_record

def training_gt_build(score_record, segment_info):
    # 使用取平均+排序的方法生成训练GT
    # 使用贪心策略找出能够使平均F1值最大化的标注作为训练GT
    keyframe_ratio=0.35
    vids = list(score_record.keys())

    # 取平均+排序
    print('Average GT: ')
    key_ratios = []
    for vid in vids:
        scores = np.array(score_record[vid]['scores'])
        scores_avg = np.mean(scores, axis=0)
        scores_list = list(scores_avg)
        scores_list.sort(reverse=True)
        threshold = scores_list[math.ceil(len(scores_avg) * keyframe_ratio)]
        labels_avg = (scores_avg > threshold).astype(int)
        score_record[vid]['scores_avg'] = scores_avg + 1e-6  # 与zero padding作区分
        score_record[vid]['labels'] = labels_avg
        key_ratios.append(np.mean(labels_avg))
        print(vid, len(scores_avg), np.mean(labels_avg))
    key_ratios = np.array(key_ratios)
    print('KeyFrame Ratio: ',keyframe_ratio,np.min(key_ratios),np.max(key_ratios),np.mean(key_ratios))

    # greedy (frame-level & segment-level)
    print('Greedy GT: ')
    for vid in vids:
        label_trues = score_record[vid]['keyshot_labels']
        savg = score_record[vid]['scores_avg']
        label_greedy_frame, f1_frame = greedy_label_build(savg, label_trues, segment_info[vid])
        score_record[vid]['label_greedy'] = label_greedy_frame
        print(vid, len(label_greedy_frame), "%.4f" % (np.sum(label_greedy_frame) / len(label_greedy_frame)), f1_frame)
    return score_record

def f1_calc(pred,gts):
    # 计算pred与所有gt的平均f1
    f1s = []
    for gt in gts:
        precision = np.sum(pred * gt) / (np.sum(pred) + 1e-6)
        recall = np.sum(pred * gt) / (np.sum(gt) + 1e-6)
        f1s.append(2 * precision * recall / (precision + recall + 1e-6))
    return np.array(f1s).mean()

def greedy_label_build(savg, keyshot_lables, cps):
    # 寻找一个可以使得与multiple annotations的F1均值最大化的标签序列，分为segment-level和frame-level两种
    # 输入scores_avg作为搜索序列

    def takescore(elem):
        return elem[1]

    # frame-level
    F1_frame = 0
    label_frame = np.zeros(len(savg))  # keyframe集合
    frame_score = list(enumerate(savg))
    frame_score.sort(key=takescore, reverse=True)  # 将（index，score）的序列排序
    while(True):
        F1_temp = F1_frame
        ind_temp = 0
        for i in range(len(frame_score)):
            f_ind, f_sc = frame_score[i]
            label_new = copy.deepcopy(label_frame)
            label_new[f_ind] = 1
            F1_new = f1_calc(label_new, keyshot_lables)
            if F1_new > F1_temp:
                F1_temp = F1_new
                ind_temp = i
            del label_new
        if F1_temp > F1_frame:
            index, _ = frame_score.pop(ind_temp)
            F1_frame = F1_temp
            label_frame[index] = 1
        else:
            break  # 新增任一帧都无法继续增大F1，则结束

    # # segment-level
    # F1_seg = 0
    # label_seg = np.zeros(len(savg))
    # while(True):
    #     F1_temp = F1_seg
    #     ind_temp = 0
    #     for i in range(len(cps) - 1):
    #         label_new = copy.deepcopy(label_seg)
    #         label_new[cps[i]:cps[i+1]] = 1
    #         F1_new = f1_calc(label_new, keyshot_lables)
    #         if F1_new > F1_temp:
    #             F1_temp = F1_new
    #             ind_temp = i
    #         del label_new
    #     if F1_temp > F1_seg:
    #         F1_seg = F1_temp
    #         label_seg[cps[ind_temp]:cps[ind_temp+1]] = 1
    #     else:
    #         break

    return label_frame, F1_frame

def max_f1_estimate(score_record, vids):
    # 使用scores_avg作为预测，计算与各个summary的F1，作为对模型可能达到的最大F1的估计
    f1_overall_greedy = []
    f1_overall_avg = []
    for vid in vids:
        label_trues = score_record[vid]['keyshot_labels']
        label_greedy = np.array(score_record[vid]['label_greedy'])
        label_avg = np.array(score_record[vid]['labels'])
        f1_greedy = f1_calc(label_greedy,label_trues)
        f1_avg = f1_calc(label_avg,label_trues)
        f1_overall_greedy.append(f1_greedy)
        f1_overall_avg.append(f1_avg)
    return np.array(f1_overall_avg), np.array(f1_overall_greedy)

if __name__ == '__main__':
    # # 只在最初运行，整合原始标注
    load_mat(GT_DIR, GT_PATH)
    print('Ground Trurth Extracted !')

    # # groundtruth summary 长度统计
    # for vid in groundtruth.keys():
    #     segs = groundtruth[vid]['segments'][0]
    #     durations = []
    #     for seg in segs:
    #         duration = 0
    #         for [start,end] in seg:
    #             duration += end - start
    #         durations.append(duration)
    #     vlength = groundtruth[vid]['video_duration'][0][0]
    #     durations = np.array(durations)
    #     print(vid,'%.4f, %.4f, %.4f' % (durations.mean() / vlength, durations.min() / vlength, durations.max() / vlength))

    # 截取帧，形成初步的score_record，只运行一次
    count = 0
    groundtruth = load_info(GT_PATH)
    score_record = {}
    for root, dirs, files in os.walk(VIDEO_BASE):
        for file in files:
            if not file.endswith('mp4'):
                continue
            vid = file.split('.mp4')[0]
            print('-'*20,count,vid,'-'*20)
            score_record[vid] = {}
            scores = frame_capture(groundtruth,vid,VIDEO_BASE,FRAME_BASE)
            # recording
            score_record[vid]['segments'] = groundtruth[vid]['segments']
            score_record[vid]['nFrames'] = groundtruth[vid]['nFrames'][0][0]
            score_record[vid]['FPS'] = groundtruth[vid]['FPS'][0][0]
            score_record[vid]['scores'] = scores
            count += 1
    with open(SCORE_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)
    print('Frames Extracted !')

    # 求测试GT
    score_record = load_info(SCORE_PATH)
    segment_info = load_info(SEGINFO_PATH)
    score_record = testing_gt_build(score_record, segment_info)
    with open(SCORE_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)

    # 求训练GT
    score_record = load_info(SCORE_PATH)
    segment_info = load_info(SEGINFO_PATH)
    score_record = training_gt_build(score_record, segment_info)
    with open(SCORE_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)
    f1_ovr_greedy = []
    f1_ovr_avg = []
    print('\n Estimating: ')
    for vid in score_record.keys():
        f1_avg, f1_greedy = max_f1_estimate(score_record,[vid])
        f1_ovr_avg.append(f1_avg)
        f1_ovr_greedy.append(f1_greedy)
        print(vid,len(score_record[vid]['label_greedy']),'Avg: %.4f, Greedy: %.4f' % (f1_avg, f1_greedy))
    f1_ovr_avg = np.array(f1_ovr_avg)
    f1_ovr_greedy = np.array(f1_ovr_greedy)
    print('Overall Estimated F1, Avg: %.4f, Greedy: %.4f' % (f1_ovr_avg.mean(), f1_ovr_greedy.mean()))
    print('Done !')
