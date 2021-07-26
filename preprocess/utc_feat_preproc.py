# 从UTC视频中提取I3D特征
# 读取视频、截取帧、裁剪、保存为数组一步完成，然后使用I3D提取特征并保存，不存储帧

import os
import numpy as np
import cv2


VIDEO_DIR = r'/data/linkang/VHL_GNN/utc/videos/'
FEATURE_DIR = r'/data/linkang/VHL_GNN/utc_feature_i3d_3fps/'

FRAME_RATE = 3
HEIGHT = 224
WIDTH = 224
CHANNEL = 3
SHOTS_NUMS = [2783, 3692, 2152, 3588]

def frame_capture(vid):
    # 每秒取3张图，15张凑一组，裁剪并保存为数组，保证shot数目与label一致
    video_path = VIDEO_DIR + 'P0%d.mp4' % (vid)
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    interval = round(fps / FRAME_RATE)  # 每隔interval帧取一帧
    rval, frame = vc.read()
    frame_count = 0
    frames = []
    while rval:
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (HEIGHT, WIDTH))
            frames.append(frame)
        if frame_count % 10000 == 0 and frame_count > 0:
            print('Frames Scanned: ',frame_count)
        rval, frame = vc.read()
        frame_count += 1
    vc.release()
    padding_num = SHOTS_NUMS[vid - 1] * FRAME_RATE * 5 - len(frames)  # 需要填充的帧数
    if padding_num < 0:
        frames = frames[: SHOTS_NUMS[vid - 1] * FRAME_RATE * 5]
    for _ in range(padding_num):
        frames.append(frames[-1])  # 使用最后一帧填充
    frames = np.array(frames).reshape([-1, 15, HEIGHT, WIDTH, CHANNEL])
    print('Vid: %s, Frames: %s, Padding: %d' % (video_path, str(frames.shape), padding_num))
    return frames

if __name__ == '__main__':
    if not os.path.isdir(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)
    for i in range(4, 5):
        print('*' * 20, 'Vid: P0%d' % i, '*' * 20)
        frames = frame_capture(i)
        np.save(FEATURE_DIR + 'P0%d_i3d_3fps.npy' % i, frames)