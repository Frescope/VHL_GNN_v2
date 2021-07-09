import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw(attention, s, b, h):
    matrix = attention[s, b, h]
    sns.heatmap(matrix, cmap='GnBu')
    plt.show()

def main():
    # attention = np.load(r'/data/linkang/model_HL_v4/attention_0_0/attention.npy')
    attention = np.load(r'/data/linkang/model_HL_v4/attention_conpred_25s/attention_conpred_25s.npy')
    # attention = np.load(r'/data/linkang/model_HL_v4/attention_conpred_35s/attention_conpred_35s.npy')
    # attention = np.load(r'/data/linkang/model_HL_v4/attention_conpred_75s/attention_conpred_75s.npy')
    seq_num, blocks, heads, sequence, _ = attention.shape
    print()

if __name__ == '__main__':
     main()