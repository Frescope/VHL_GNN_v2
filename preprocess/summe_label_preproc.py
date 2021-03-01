import scipy.io
import os
import json
import numpy as np

GT_DIR = r'/data/linkang/SumMe/GT/'
GT_PATH = r'/data/linkang/SumMe/groundtruth.json'

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
            vid = name.split('mat')[0]
            groundtruth[vid] = {}
            groundtruth[vid]['all_userIDs'] = gt_data['all_userIDs']
            groundtruth[vid]['segments'] = gt_data['segments']
            groundtruth[vid]['nFrames'] = gt_data['nFrames']
            groundtruth[vid]['video_duraton'] = gt_data['video_duration']
            groundtruth[vid]['FPS'] = gt_data['FPS']
            groundtruth[vid]['gt_score'] = gt_data['gt_score']
            groundtruth[vid]['user_score'] = gt_data['user_score']
    with open(gt_path, 'w') as file:
        json.dump(groundtruth, file, cls=NpEncoder)
    print('Done !')

def load_gt(gt_path):
    with open(gt_path, 'r') as file:
        groundtruth = json.load(file)
    return groundtruth



if __name__ == '__main__':
    # load_mat(GT_DIR, GT_PATH)
    groundtruth = load_gt(GT_PATH)
    print()