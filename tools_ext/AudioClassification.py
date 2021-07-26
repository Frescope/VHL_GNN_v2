"""
来源：https://github.com/qiuqiangkong/audioset_classification
Kong, Qiuqiang, Changsong Yu, Yong Xu, Turab Iqbal, Wenwu Wang, and Mark D. Plumbley. "Weakly Labelled AudioSet Tagging With Attention Neural Networks." IEEE/ACM Transactions on Audio, Speech, and Language Processing 27, no. 11 (2019): 1791-1802.

一种基于音频的Vggish特征进行分类的模型；原文中使用的是来自AudioSet数据集的输入，每个样例的长度固定，均为(10,128)的形式，其中10为audio_frame数量。
模型输入的形式是(bs,N,d)，需要对不同长度的输入特征（即N不同）做padding；模型输出(bs,C)C为对输出的类别数；注意bs=1可能会影响模型内做BatchNorm的步骤。
uint8_to_float32,bool_to_float32为格式转换函数，init_layer，init_bn分别为卷积层BatchNorm的初始化操作，与其余函数均为模型组件，使用FeatureLevelSingleAttention调用模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np
import math
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = '8'

path = r'/data/linkang/audioset/packed_features/bal_train.h5'
with h5py.File(path, 'r') as hf:
    X = hf['x'][:]
    y = hf['y'][:]
    video_id_list = hf['video_id_list'][:].tolist()

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)

class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        # 三次全连接
        self.conv1 = nn.Conv2d(
            in_channels=freq_bins, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv3 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.bn0 = nn.BatchNorm2d(freq_bins)
        self.bn1 = nn.BatchNorm2d(hidden_units)
        self.bn2 = nn.BatchNorm2d(hidden_units)
        self.bn3 = nn.BatchNorm2d(hidden_units)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

        init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins) = (batchsize, frame_num, 128)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        x = x[:, :, :, None].contiguous()

        a0 = self.bn0(x)
        a1 = F.dropout(F.relu(self.bn1(self.conv1(a0))),
                       p=drop_rate,
                       training=self.training)

        a2 = F.dropout(F.relu(self.bn2(self.conv2(a1))),
                       p=drop_rate,
                       training=self.training)

        emb = F.dropout(F.relu(self.bn3(self.conv3(a2))),
                        p=drop_rate,
                        training=self.training)

        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return emb

        else:
            return [a0, a1, a2, emb]

class Attention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):  # n_in = n_out = hidden_units
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

    def init_weights(self):
        init_layer(self.att, )
        init_layer(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, hidden_units, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]  # (samples_num, hidden_units, time_steps)
        cla = cla[:, :, :, 0]  # (samples_num, hidden_units, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        return x

class FeatureLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):
        super(FeatureLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)

        self.attention = Attention(
            hidden_units,
            hidden_units,
            att_activation='sigmoid',
            cla_activation='linear')

        self.fc_final = nn.Linear(hidden_units, classes_num)
        self.bn_attention = nn.BatchNorm1d(hidden_units)

        self.drop_rate = drop_rate

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)
        init_bn(self.bn_attention)

    def forward(self, input):
        """input: (samples_num, time_steps, freq_bins)
        """
        drop_rate = self.drop_rate

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, hidden_units)
        b2 = self.attention(b1)
        b2 = F.dropout(
            F.relu(
                self.bn_attention(b2)),
            p=drop_rate,
            training=self.training)

        # (samples_num, classes_num)
        output = F.sigmoid(self.fc_final(b2))

        return output

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.

def bool_to_float32(y):
    return np.float32(y)

# training test
def train():
    freq_bins = 128  # Vggish特征长度
    classes_num = 527  # AudioSet中共有527个类别
    hidden_units = 1024
    drop_rate = 0.2
    bs = 10

    model = FeatureLevelSingleAttention(freq_bins, classes_num, hidden_units, drop_rate)
    model.cuda()  # Move model to gpu

    # Optimization method
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-07)

    for i in range(100):
        # 每次输入一个样本，样本长度可能不固定，输入多个时需要padding
        batch_x = X[i * bs : (i+1) * bs]
        batch_y = y[i * bs : (i+1) * bs]
        batch_x = uint8_to_float32(batch_x)  # Vggish特征是uint8格式
        batch_y = bool_to_float32(batch_y)  # 标签需要转化为浮点形式

        batch_x = Variable(torch.Tensor(batch_x).cuda())
        batch_y = Variable(torch.Tensor(batch_y).cuda())

        model.train()
        output = model(batch_x)
        loss = F.binary_cross_entropy(output, batch_y)

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i, output.shape)

if __name__ == '__main__':
    train()

