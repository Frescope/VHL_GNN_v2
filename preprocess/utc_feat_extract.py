# 使用sonnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tools_ext.i3d as i3d
import os
import math
import numpy as np

DATA_DIR = r'/data/linkang/VHL_GNN/utc_feature_i3d_3fps/'
FEATURE_DIR = r'/data/linkang/VHL_GNN/utc/i3d_features/'

os.environ["CUDA_VISIBLE_DEVICES"] = '8'
_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 15
NUM_CLASSES = 600
BATCH_SIZE = 20

SHOTS_NUMS = [2783, 3692, 2152, 3588]

_CHECKPOINT_PATHS = {
    'rgb': '/data/linkang/PycharmProjects/i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': '/data/linkang/PycharmProjects/i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': '/data/linkang/PycharmProjects/i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '/data/linkang/PycharmProjects/i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '/data/linkang/PycharmProjects/i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

def getBatch(feature, step, bc):
    start = step * bc
    end = (step + 1) * bc
    feature_b = feature[start : end]
    padding_len = end - len(feature)
    if padding_len > 0:
        feature_pad = np.zeros((padding_len, 15, 224, 224, 3))
        feature_b = np.vstack((feature_b, feature_pad))
    return  feature_b

def run(rgb_data):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = 'rgb600'

    rgb_input = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            # rgb_variable_map[variable.name.replace(':0', '')] = variable
            rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    with tf.Session() as sess:
        feed_dict = {}
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
        tf.logging.info('RGB checkpoint restored')
        tf.logging.info('RGB data loaded, shape=%s', str(len(rgb_data)))

        max_step = math.ceil(len(rgb_data) / BATCH_SIZE)
        i3d_features = []
        for step in range(max_step):
            feature_batch = getBatch(rgb_data,step,BATCH_SIZE)
            feed_dict[rgb_input] = feature_batch
            out_logits = sess.run(rgb_logits, feed_dict=feed_dict)
            # print(step, out_logits.shape)
            i3d_features.append(out_logits)
        i3d_features = np.array(i3d_features).reshape((-1, NUM_CLASSES))
        return i3d_features

def main(self):
    # if not os.path.isdir(FEATURE_DIR):
    #     os.makedirs(FEATURE_DIR)
    # for i in range(4, 5):
    #     print('*' * 20, 'Vid: P0%d' % i, '*' * 20)
    #     data_path = DATA_DIR + 'P0%d_i3d_3fps.npy' % i
    #     feature_path = FEATURE_DIR + 'V%d_I3D.npy' % i
    #     rgb_data = np.load(data_path)
    #     features = run(rgb_data)
    #     np.save(feature_path, features[: len(rgb_data)])
    #     print('Extracted: ', features.shape)

    # temp
    for i in range(1, 5):
        feature_path = FEATURE_DIR + 'V%d_I3D.npy' % i
        feature = np.load(feature_path)
        feature = feature[: SHOTS_NUMS[i - 1]]
        print(feature.shape)
        np.save(FEATURE_DIR + 'V%d_I3D_2.npy' % i, feature)

if __name__ == '__main__':
    tf.app.run(main)


