# 使用CLIP提取特征

import clip
import torch
import os
import json
import pickle
import numpy as np
from PIL import Image

D_OUTPUT = 512

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

def load_info(info_path):
    with open(info_path, 'r') as file:
        info = json.load(file)
        return info

def getVisualFeatures(frame_dir, feature_base):
    if not os.path.isdir(feature_base):
        os.makedirs(feature_base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    with torch.no_grad():
        for vid in os.listdir(frame_dir):
            print(vid, len(os.listdir(os.path.join(frame_dir, vid))))
            image_list = []
            visual_features = []
            for frame in os.listdir(os.path.join(frame_dir, vid)):
                frame_path = os.path.join(frame_dir, vid, frame)
                image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
                image_list.append(image)
                if len(image_list) >= 200:
                    images = torch.cat(image_list, 0)
                    image_list.clear()
                    image_feature = model.encode_image(images)
                    visual_features.append(image_feature.cpu().numpy())
            images = torch.cat(image_list, 0)
            image_list.clear()
            image_feature = model.encode_image(images)
            visual_features.append(image_feature.cpu().numpy())

            vid_features = np.concatenate(visual_features, axis=0)
            visual_features.clear()
            np.save(feature_base + '%s_CLIP_2fps.npy' % vid, vid_features)
            print('CLIP Features: ', vid, vid_features.shape)
    return

def getTextFeatures(dict_path, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    if dataset == 'tvsum':
        dict = {
            'VT': 'Changing Vehicle Tire',
            'VU': 'Getting Vehicle Unstuck',
            'GA': 'Grooming an Animal',
            'MS': 'Making Sandwich',
            'PK': 'Parkour',
            'PR': 'Parade',
            'FM': 'Flash Mob Gathering',
            'BK': 'Bee Keeping',
            'BT': 'Attempting Bike Tricks',
            'DS': 'Dog Show'
        }
        for category in dict:
            text = dict[category]
            dict[category] = {}
            with torch.no_grad():
                text_emb = clip.tokenize(text).to(device)
                text_feature = model.encode_text(text_emb)
                dict[category]['text'] = text
                dict[category]['feature'] = text_feature.cpu().numpy()
    elif dataset == 'summe':
        names = os.listdir(r'/data/linkang/SumMe/frame_2fps/')
        dict = {}
        for vname in names:
            dict[vname] = {}
            text = ' '.join(vname.split('_'))
            with torch.no_grad():
                text_emb = clip.tokenize(text).to(device)
                text_feature = model.encode_text(text_emb)
                dict[vname]['text'] = text
                dict[vname]['feature'] = text_feature.cpu().numpy()
    else:
        cates = {
            'base': 'Base jumping',
            'bike': 'Bike Polo',
            'eiffel': 'Eiffel Tower',
            'exca': 'Excavators river crossing',
            'kids': 'Kids playing in leaves',
            'mlb': 'Major League Baseball',
            'nfl': 'National Football League',
            'notre': 'Notre Dame',
            'statue': 'Statue of Liberty',
            'surf': 'Surfing'
        }
        dict = {}
        for name in cates:
            dict[name] = {}
            text = cates[name]
            with torch.no_grad():
                text_emb = clip.tokenize(text).to(device)
                text_feature = model.encode_text(text_emb)
                dict[name]['text'] = text
                dict[name]['feature'] = text_feature.cpu().numpy()
    with open(dict_path, 'wb') as file:
        pickle.dump(dict, file)
    return

def CLIP_tvsum():
    FRAME_DIR = r'/data/linkang/tvsum50/frame_2fps/'
    FEATURE_BASE = r'/data/linkang/tvsum50/feature_clip_2fps/'
    CONCEPT_PATH = r'/data/linkang/tvsum50/concept_clip.pkl'
    ANNO_PATH = r'/data/linkang/tvsum50/anno_info.json'
    VIDEO_INFO_PATH = r'/data/linkang/tvsum50/video_info.json'
    TEXT_DICT_PATH = r'/data/linkang/Uniset/tvsum_clip_text.pkl'

    # getVisualFeatures(FRAME_DIR, FEATURE_BASE)
    getTextFeatures(TEXT_DICT_PATH, 'tvsum')

def CLIP_summe():
    FRAME_DIR = r'/data/linkang/SumMe/frame_2fps/'
    FEATURE_BASE = r'/data/linkang/Uniset/summe_clip_visual_2fps/'
    GT_PATH = r'/data/linkang/Uniset/summe_score_record.json'
    TEXT_DICT_PATH = r'/data/linkang/Uniset/summe_clip_text.pkl'

    # gt = load_info(GT_PATH)
    # for vname in os.listdir(FRAME_DIR):
    #     print(vname, len(os.listdir(FRAME_DIR + vname)), len(gt[vname]['scores_avg']))

    # getVisualFeatures(FRAME_DIR, FEATURE_BASE)
    getTextFeatures(TEXT_DICT_PATH, 'summe')

def CLIP_cosum():
    FRAME_DIR = r'/data/linkang/CoSum/frame_2fps/'
    FEATURE_BASE = r'/data/linkang/Uniset/cosum_clip_visual_2fps/'
    TEXT_DICT_PATH = r'/data/linkang/Uniset/cosum_clip_text.pkl'

    # getVisualFeatures(FRAME_DIR, FEATURE_BASE)
    getTextFeatures(TEXT_DICT_PATH, 'cosum')

def label_build():
    # tvsum与summe的原始标签是对应到每一帧的、每个标注者的评分
    # cosum的原始标签是三种类型的、对应到每一帧的二元标注
    # 需要测试ranking & f1两种指标
    # 对f1指标：
    #   设定一个比例作为门限，对每组人工标注，选择得分最高的部分帧标记为1，其余为0，作为多标签ref；
    #   对所有得分取均值，再选择得分最高的部分标记为1，其余为0，作为单标签ref；
    #   使用单标签ref作为训练标签；
    #   对cosum使用dualplus作为训练标签和测试标签。
    # 对ranking指标：
    #   同样对tvsum与summe得到多标签ref和单标签ref，区别是不映射到二元标签上；
    #   cosum上同样使用dualplus；
    #   将tvsum与summe的单标签ref做0-1标准化，作为训练标签。

    def min_max(scores):
        # 对若干组得分做0-1标准化, N * vlength
        min_values = np.min(scores, axis=1, keepdims=True)
        max_values = np.max(scores, axis=1, keepdims=True)
        result = (scores - min_values) / (max_values - min_values + 1e-8)
        return result

    def score2binary(scores):
        # 按照指定比例将连续标签转化为二元标签
        hlnum = int(len(scores[0]) * BINARY_RATIO)
        binary_list = []
        for score in scores:
            scores_list = list(score)
            scores_list.sort(reverse=True)
            threshold = scores_list[hlnum]
            binary_list.append((score > threshold).astype(float))
        return np.array(binary_list)

    TVSUM_LABEL_PATH = r'/data/linkang/Uniset/labels_raw/tvsum_score_record.json'
    SUMME_LABEL_PATH = r'/data/linkang/Uniset/labels_raw/summe_score_record.json'
    COSUM_LABEL_PATH = r'/data/linkang/Uniset/labels_raw/cosum_labels.json'
    UNISET_LABEL_PATH = r'/data/linkang/Uniset/uniset_labels.json'
    tvsum_label = load_info(TVSUM_LABEL_PATH)
    summe_label = load_info(SUMME_LABEL_PATH)
    cosum_label = load_info(COSUM_LABEL_PATH)
    BINARY_RATIO = 0.18

    # tvsum
    tvsum_dict = {}
    for vid in tvsum_label:
        scores = np.array(tvsum_label[vid]['scores'])  # N * vlength
        scores_sum = np.sum(scores, axis=0, keepdims=True)  # 1 * vlength
        scores = min_max(scores)  # 连续多标签
        scores_single = min_max(scores_sum)  # 连续单标签
        binary = score2binary(scores)  # 二元多标签
        binary_single = score2binary(scores_single)  # 二元单标签
        tvsum_dict[vid] = {}
        tvsum_dict[vid]['multi_score'] = scores
        tvsum_dict[vid]['single_score'] = scores_single
        tvsum_dict[vid]['multi_binary'] = binary
        tvsum_dict[vid]['single_binary'] = binary_single

    # summe
    summe_dict = {}
    for vid in summe_label:
        scores = np.array(summe_label[vid]['scores'])  # N * vlength
        scores_sum = np.sum(scores, axis=0, keepdims=True)  # 1 * vlength
        scores = min_max(scores)  # 连续多标签
        scores_single = min_max(scores_sum)  # 连续单标签
        binary = score2binary(scores)  # 二元多标签
        binary_single = score2binary(scores_single)  # 二元单标签
        summe_dict[vid] = {}
        summe_dict[vid]['multi_score'] = scores
        summe_dict[vid]['single_score'] = scores_single
        summe_dict[vid]['multi_binary'] = binary
        summe_dict[vid]['single_binary'] = binary_single

    # cosum
    cosum_dict = {}
    for vid in cosum_label:
        cosum_dict[vid] = {}
        cosum_dict[vid]['dualplus'] = cosum_label[vid]['dualplus']

    # Uniset
    Uniset = {
        'tvsum': tvsum_dict,
        'summe': summe_dict,
        'cosum': cosum_dict
    }
    with open(UNISET_LABEL_PATH, 'w') as file:
        json.dump(Uniset, file, cls=NpEncoder)
    return

if __name__ == '__main__':
    # CLIP_tvsum()
    # CLIP_summe()
    # CLIP_cosum()
    # Uniset = label_build()

    with open(r'/data/linkang/Uniset/tvsum_clip_text.pkl', 'rb') as file:
        tvsum = pickle.load(file)
    with open(r'/data/linkang/Uniset/summe_clip_text.pkl', 'rb') as file:
        summe = pickle.load(file)
    with open(r'/data/linkang/Uniset/cosum_clip_text.pkl', 'rb') as file:
        cosum = pickle.load(file)
    with open(r'/data/linkang/Uniset/uniset_labels.json', 'r') as file:
        uniset = json.load(file)

    print()

