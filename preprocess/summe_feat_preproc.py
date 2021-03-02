# summe特征与标签的前处理，包括帧的截取与标签的制作，后续只进行特征提取即可
# 读取GT，抽取groundtruth
# 截取帧，获取对应的标注形成score_record，用贪心策略求出训练GT

import os
import numpy as np
import cv2
import math
import scipy.io
import json
import copy

VIDEO_BASE = r'/data/linkang/SumMe/videos/'
FRAME_BASE = r'/data/linkang/SumMe/frame_2fps/'
GT_DIR = r'/data/linkang/SumMe/GT/'
GT_PATH = r'/data/linkang/SumMe/groundtruth.json'
SCORE_PATH = r'/data/linkang/VHL_GNN/summe_score_record.json'
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
    # if os.path.isdir(frame_dir):
    #     print('Directory already exists !')
    #     return
    # else:
    #     os.makedirs(frame_dir)

    # get annotations
    user_score = np.array(groundtruth[vid]['user_score'])  # 关于这一id的所有标注
    nFrames = groundtruth[vid]['nFrames'][0][0]
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
            # cv2.imwrite(frame_path, frame)
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

def training_gt_build(score_record):
    # 找出能够使平均F1值最大化的标注作为训练GT
    vids = list(score_record.keys())
    for vid in vids:
        print('-'*20,vid,'-'*20)
        label_trues = score_record[vid]['scores']
        savg = np.mean(np.array(label_trues),axis=0).reshape(-1)
        label_greedy, f1 = greedy_label_build(savg, label_trues)
        score_record[vid]['label_greedy'] = label_greedy
        print(vid, len(label_greedy), np.sum(label_greedy), "%.4f" % (np.sum(label_greedy) / len(label_greedy)), f1)
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

def max_f1_estimate(score_record, vids):
    # 使用scores_avg作为预测，计算与各个summary的F1，作为对模型可能达到的最大F1的估计
    f1_overall_greedy = []
    for vid in vids:
        label_trues = score_record[vid]['scores']
        label_greedy = np.array(score_record[vid]['label_greedy'])
        f1_greedy = f1_calc(label_greedy,label_trues)
        f1_overall_greedy.append(f1_greedy)
    return np.array(f1_overall_greedy)

if __name__ == '__main__':
    # 只在最初运行，整合原始标注
    load_mat(GT_DIR, GT_PATH)
    print('Ground Trurth Extracted !')

    # 截取帧，形成初步的score_record，只运行一次
    count = 0
    groundtruth = load_info(GT_PATH)
    # summary 长度统计
    for vid in groundtruth.keys():
        segs = groundtruth[vid]['segments'][0]
        durations = []
        for seg in segs:
            duration = 0
            for [start,end] in seg:
                duration += end - start
            durations.append(duration)
        vlength = groundtruth[vid]['video_duration'][0][0]
        durations = np.array(durations)
        print(vid,'%.4f, %.4f, %.4f' % (durations.mean() / vlength, durations.min() / vlength, durations.max() / vlength))

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


    # 用贪心策略求训练GT
    score_record = load_info(SCORE_PATH)
    # score_record = training_gt_build(score_record)
    with open(SCORE_PATH, 'w') as file:
        json.dump(score_record, file, cls=NpEncoder)
    for vid in score_record.keys():
        f1 = max_f1_estimate(score_record,[vid])
        label_trues = np.array(score_record[vid]['scores'])
        mean_ratio = np.mean(np.sum(label_trues,axis=1)) / label_trues.shape[1]
        print(vid,label_trues.shape[1],'%.3f' % mean_ratio, f1)

    print('Done !')
