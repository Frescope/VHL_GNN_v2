# 读取视频，按照固定帧率截取图像，保存为jpg格式，对图像大小不作要求，统一留到特征提取时处理

import os
import numpy as np
import csv
import cv2
import math
import json

VIDEO_BASE = r'/data/linkang/tvsum50/video/'
FRAME_BASE = r'/data/linkang/tvsum50/frame_2fps/'
FRAME_RATE = 2  # 2fps

def frame_capture(vid,video_base,frame_base):
    video_path = video_base + vid + '.mp4'
    frame_dir = frame_base + vid + r'/'
    if os.path.isdir(frame_dir):
        print('Directory already exists !')
        return
    else:
        os.makedirs(frame_dir)

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_interval = round(fps / FRAME_RATE)  # 截取帧的间隔
    # capturing
    rval, frame = vc.read()
    frame_count = 0
    while rval:
        if frame_count % frame_interval == 0:
            frame_time = round(frame_count / fps * 10)  # 帧对应的秒数*10
            frame_path = frame_dir + str(frame_time).zfill(6) + '.jpg'
            cv2.imwrite(frame_path, frame)
        if frame_count % 500 == 0 and frame_count > 0:
            print('Frames: ',frame_count)
        rval, frame = vc.read()
        frame_count += 1
    vc.release()
    print('Frames Extracted: ',vid,frame_count)

if __name__ == '__main__':
    count = 0
    for root, dirs, files in os.walk(VIDEO_BASE):
        for file in files:
            vid = file.split('.mp4')[0]
            print('-'*20,count,vid,'-'*20)
            frame_capture(vid,VIDEO_BASE,FRAME_BASE)
    print('Done !')