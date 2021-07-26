# 从UTC视频中截取帧，1fps

import os
import cv2
import math

VIDEO_BASE = r'/data/linkang/VHL_GNN/utc/videos/'
FRAME_DIR = r'/data/linkang/VHL_GNN/utc/frames/'
SHOTS_NUMS = [2783, 3692, 2152, 3588]

def frame_cap(video_base, vid, frame_dir):
    # 每隔若干帧取一帧，按对应的秒数取下整保存
    video_path = video_base + 'P0%d.mp4' % vid
    frame_base = frame_dir + 'P0%d/' % vid
    if not os.path.isdir(frame_base):
        os.makedirs(frame_base)

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    interval = math.floor(fps)

    rval, frame = vc.read()
    count = 0
    cap_num = 0
    while rval:
        if count % interval == 0:
            second = math.floor(count / fps)  # 秒数
            path = frame_base + str(second).zfill(5) + '.jpg'
            cv2.imwrite(path, frame)
            cap_num += 1
            if cap_num % 1000 == 0 and cap_num > 0:
                print('Frames: ', count, cap_num)
        rval, frame = vc.read()
        count += 1
    vc.release()
    print('Frame Captured: ', vid, count, cap_num)

    return

def frame_check(frame_dir, vid):
    # 检查并补充缺少的帧
    frame_base = frame_dir + 'P0%d/' % vid
    for sec in range(SHOTS_NUMS[vid - 1] * 5):
        path = frame_base + str(sec).zfill(5) + '.jpg'
        if not os.path.isfile(path):
            # 寻找上一帧
            sec_pre = sec - 1
            while not os.path.isfile(frame_base + str(sec_pre).zfill(5) + '.jpg'):
                sec_pre -= 1
            print(path, sec_pre)
            os.system('cp %s %s' % (
                frame_base + str(sec_pre).zfill(5) + '.jpg',
                frame_base + str(sec).zfill(5) + '.jpg')
            )

    return

if __name__ == '__main__':
    # for i in range(1, 5):
    #     frame_cap(VIDEO_BASE, i, FRAME_DIR)

    for i in range(1, 5):
        frame_check(FRAME_DIR, i)