import os
import scipy.io

ANNOTATION_DIR = r'/data/linkang/CoSum/annotation/'
SEGMENT_DIR = r'/data/linkang/CoSum/shots/'

CATEGORIES = {
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

def load_annotations():
    annotations = {}
    for dir in os.listdir(ANNOTATION_DIR):
        if not os.path.isdir(os.path.join(ANNOTATION_DIR, dir)):
            continue
        cate_id = int(dir.split('_')[0])
        if cate_id not in annotations:
            annotations[cate_id] = {}
        for file in os.listdir(os.path.join(ANNOTATION_DIR, dir)):
            v_num, type = file.split('.mat')[0].split('__')
            label = scipy.io.loadmat(os.path.join(ANNOTATION_DIR, dir, file))['labels']
            if v_num not in annotations[cate_id]:
                annotations[cate_id][v_num] = {}
            annotations[cate_id][v_num][type] = label
            print(dir, v_num, type, label.shape)
    return annotations

def load_segment():
    segment_info = {}
    for dir in os.listdir(SEGMENT_DIR):
        if not os.path.isdir(os.path.join(SEGMENT_DIR, dir)):
            continue
        cate_id = int(dir.split('_')[0])
        if cate_id not in segment_info:
            segment_info[cate_id] = {}
        for file_name in os.listdir(os.path.join(SEGMENT_DIR, dir)):
            v_num = file_name.split('_')[0][-1]
            with open(os.path.join(SEGMENT_DIR, dir, file_name)) as file:
                indices = []
                for line in file.readlines():
                    indices.append(int(line))
            segment_info[cate_id][v_num] = indices
            print(cate_id, v_num, len(indices))
    return segment_info

def check_annotations(annotations, segment_info):
    for cate_id in annotations:
        for v_num in annotations[cate_id]:
            # shots = {}
            # for type in annotations[cate_id][v_num]:
            #     shots[type] = list(annotations[cate_id][v_num][type][:, 0])
            # union_set = set(shots['kk']) | set(shots['vv'])
            # print(len(shots['dualplus']), len(shots['kk']), len(shots['vv']), len(union_set))
            dualplus_shots = list(annotations[cate_id][v_num]['dualplus'][:, 0])
            shots_num = len(segment_info[cate_id][v_num]) - 1
            print(cate_id, CATEGORIES[cate_id], v_num, len(dualplus_shots), shots_num,
                  '%.2f' % (len(dualplus_shots) / shots_num))

annos = load_annotations()
segment_info = load_segment()
check_annotations(annos, segment_info)