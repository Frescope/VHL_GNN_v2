# 检查具有query标签的shot所占的比例
import os

FEATURE_BASE = r'/data/linkang/VHL_GNN/utc/origin_data/Query-Focused_Summaries/Oracle_Summaries/'

for i in [1,2,3,4]:
    labeled_shots = []
    feature_dir = FEATURE_BASE + 'P0%d/' % i
    for root, dirs, files in os.walk(feature_dir):
        for name in files:
            with open(os.path.join(root, name)) as file:
                lines = file.readlines()
                for line in lines:
                    labeled_shots.append(int(line.strip()) - 1)
    labeled_shots = list(set(labeled_shots))
    labeled_shots.sort()
    print('Vid: %d Shots: %d, Length: %d Ratio: %.3f' %(i, len(labeled_shots), max(labeled_shots),
                                                        len(labeled_shots) / max(labeled_shots)))