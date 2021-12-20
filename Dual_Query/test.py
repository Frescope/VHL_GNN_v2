import os
import json
from scipy import stats
import numpy as np

uniset_base = r'/data/linkang/Uniset/'

# with open(uniset_base + 'uniset_labels.json', 'r') as file:
#     uniset_labels = json.load(file)
# with open(uniset_base + 'labels_raw/tvsum_score_record.json', 'r') as file:
#     tvsum_score_record = json.load(file)
#
# t1 = uniset_labels['tvsum']
# t2 = tvsum_score_record
# print()
# for vid in t1:
#     l1 = t1[vid]['single_binary'][0]
#     l2 = t2[vid]['labels']
#     print(vid, sum(l1) / len(l1), sum(l2) / len(l2))

ks = []
ss = []
N = 5
for i in range(100):
    l1 = np.random.random((N))
    l2 = np.array(range(N))
    k, _ = stats.kendalltau(l1, l2)
    s, _ = stats.spearmanr(l1, l2)
    ks.append(k)
    ss.append(s)
    print(i, k, s)
print(sum(ks) / len(ks), sum(ss) / len(ss))