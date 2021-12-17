# 对融合数据集的帧捕捉、图像与文本的CLIP特征提取
# 包括TVSum，SumMe，CoSum三个数据集，按照每秒2帧抽取帧，使用CLIP提取图像特征
# 所有视频根据其所属类型或主题，使用CLIP抽取文本特征
# 注意区分不同数据集的特征

import os
import cv2
import math
import json
import scipy.io
import numpy as np

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

# for tvsum

# for summe

# for cosum

def load_annotations(anno_base, seg_base, video_dir):
    # 加载标签以及分段信息，形成frame-level标注
    cate_map = {
        1: 'base',
        2: 'bike',
        3: 'eiffel',
        4: 'exca',
        5: 'kids',
        6: 'mlb',
        7: 'nfl',
        8: 'notre',
        9: 'statue',
        10: 'surf'
    }

    # load annotations
    anno_dict = {}
    for anno_dir in os.listdir(anno_base):
        if not os.path.isdir(os.path.join(anno_base, anno_dir)):
            continue
        category = cate_map[int(anno_dir.split('_')[0])]
        for file in os.listdir(os.path.join(anno_base, anno_dir)):
            num, type = file.split('.mat')[0].split('__')
            label = scipy.io.loadmat(os.path.join(anno_base, anno_dir, file))['labels']  # shot索引是从1开始的
            if category+num not in anno_dict:
                anno_dict[category+num] = {}
            anno_dict[category+num][type] = label

    # load segment_info
    seg_dict = {}
    for seg_dir in os.listdir(seg_base):
        if not os.path.isdir(os.path.join(seg_base, seg_dir)):
            continue
        for file in os.listdir(os.path.join(seg_base, seg_dir)):
            cate = file.split('_')[0]
            with open(os.path.join(seg_base, seg_dir, file)) as file:
                indices = []
                for line in file.readlines():
                    indices.append(int(line))  # anno中的shot索引n对应n行和n-1行之间的帧
            seg_dict[cate] = indices

    # load video info
    video_dict = {}
    for vname in os.listdir(video_dir):
        vc = cv2.VideoCapture(os.path.join(video_dir, vname))
        vlabel = vname.split('-')[0]
        video_dict[vlabel] = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    # build frame-level label
    frame_level_label = {}
    for key in anno_dict:
        if key not in video_dict:
            continue
        frame_num = int(video_dict[key])
        labels = {
            'dualplus': [],
            'kk':  [],
            'vv': [],
        }
        for type in anno_dict[key]:
            label_type = np.zeros(frame_num)
            for i in range(len(anno_dict[key][type]) - 1):
                st = seg_dict[key][anno_dict[key][type][i][0] - 1]
                ed = seg_dict[key][anno_dict[key][type][i][0]]
                label_type[st:ed] = 1
            labels[type] = label_type
        frame_level_label[key] = labels
    return frame_level_label

def frame_capture_cosum():
    # 截取帧，并记录对应帧的标签
    def capture(video_path, frame_dir, vlabel):
        vc = cv2.VideoCapture(video_path)
        fps = vc.get(cv2.CAP_PROP_FPS)

        rval, frame = vc.read()
        count = 0
        cap_num = 0
        frame_list = []  # 捕捉的帧序号
        while rval:
            time = int(round(count / fps, 1) * 10)  # 秒数乘10
            if time % 5 == 0:  # 每秒取2帧
                path = frame_dir + '/' + str(time).zfill(5) + '.jpg'
                if not os.path.isfile(path):
                    cv2.imwrite(path, frame)
                    cap_num += 1
                    frame_list.append(count)
                    if cap_num % 1000 == 0 and cap_num > 0:
                        print('Frames: ', count, cap_num)
            rval, frame = vc.read()
            count += 1
        vc.release()
        print('Frame Captured: ', vlabel, count, cap_num)
        return frame_list, fps, count

    VIDEO_DIR = r'/data/linkang/CoSum/videos/'
    FRAME_BASE = r'/data/linkang/CoSum/frame_2fps/'
    ANNO_BASE = r'/data/linkang/CoSum/annotation/'
    SEG_BASE = r'/data/linkang/CoSum/shots/'
    LABEL_PATH = r'/data/linkang/CoSum/labels.json'

    frame_level_label = load_annotations(ANNO_BASE, SEG_BASE, VIDEO_DIR)

    labels = {}
    for vname in os.listdir(VIDEO_DIR):
        vlabel = vname.split('-')[0]
        frame_dir = os.path.join(FRAME_BASE, vlabel)
        if not os.path.isdir(frame_dir):
            os.makedirs(frame_dir)
        video_path = os.path.join(VIDEO_DIR, vname)
        vlabel = vname.split('-')[0]  # 用于索引info
        flist, fps, fcount = capture(video_path, frame_dir, vlabel)

        labels[vlabel] = {}
        for type in frame_level_label[vlabel]:
            labels[vlabel][type] = frame_level_label[vlabel][type][flist]
            print(vlabel, type, labels[vlabel][type].shape)
        labels[vlabel]['frame_list'] = flist
        labels[vlabel]['frame_count'] = fcount
        labels[vlabel]['fps'] = fps

        with open(r'/data/linkang/CoSum/temp_labels/'+vlabel+'.json', 'w') as file:
            json.dump(labels[vlabel], file, cls=NpEncoder)

    print('Frames Captured !')

    with open(LABEL_PATH, 'w') as file:
        json.dump(labels, file, cls=NpEncoder)
    return

frame_capture_cosum()
