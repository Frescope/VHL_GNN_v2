import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    attention = np.load(r'/data/linkang/model_HL_v4/attention_0_0/attention.npy')
    seq_num, blocks, heads, sequence, _ = attention.shape
    matrix = attention[0, 5, 4]
    sns.heatmap(matrix, cmap='GnBu')
    plt.show()

if __name__ == '__main__':
     main()