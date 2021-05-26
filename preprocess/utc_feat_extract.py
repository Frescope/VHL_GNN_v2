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

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 15
NUM_CLASSES = 400
BATCH_SIZE = 1

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

def run(data):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = 'rgb_imagenet'

    rgb_input = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        _, end_points = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_logits = end_points['Logits']
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
            # rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    with tf.Session() as sess:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
        tf.logging.info('RGB checkpoint restored')
        features = {}
        for vid in data:
            feed_dict = {}
            rgb_data = data[vid]
            tf.logging.info('RGB data loaded, shape=%s', str(len(rgb_data)))

            max_step = math.ceil(len(rgb_data) / BATCH_SIZE)
            i3d_features = []
            for step in range(max_step):
                feature_batch = getBatch(rgb_data,step,BATCH_SIZE)
                feed_dict[rgb_input] = feature_batch
                out_logits = sess.run(rgb_logits, feed_dict=feed_dict)
                # print(step, out_logits.shape)
                i3d_features.append(out_logits)
            i3d_features = np.array(i3d_features).reshape((-1, 1024))
            features[vid] = i3d_features
        return features

def main(self):
    if not os.path.isdir(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)
    data = {}
    for i in range(1, 5):
        data_path = DATA_DIR + 'P0%d_i3d_3fps.npy' % i
        rgb_data = np.load(data_path)
        print('Vid %d: ' % i, rgb_data.shape)
        data[str(i)] = rgb_data
    features = run(data)
    for vid in features:
        print('*' * 20, 'Vid: P0%d' % i, '*' * 20)
        feature_path = FEATURE_DIR + 'V%s_I3D.npy' % vid
        np.save(feature_path, features[vid][: SHOTS_NUMS[int(vid) - 1]])
        print('Extracted: ', vid, features[vid].shape)

if __name__ == '__main__':
    tf.app.run(main)


