#!/usr/bin/env python

import os
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot




src_file = 'test_video.mp4'
temp_dir = 'res_me/'
dst_file = 'res_me/out_video_0.4.mp4'
v_in = mmcv.VideoReader(src_file)
# v_out = mmcv.video.VideoWriter(dst_file,v_in.fps)
v_in.cvt2frames(temp_dir)

confidence_threshold = 0.4
config_file = "train.py"
checkpoint_file = "res_me/latest.pth"
model = init_detector(config_file, checkpoint_file)

for file in os.listdir(temp_dir):
    file = os.path.join(temp_dir, file)
    print(file)
    if not file.endswith(".jpg"):
        continue
    frame = mmcv.imread(file)
    result = inference_detector(model, frame)

    with open('res_me/rest.txt', 'w') as f:
        f.write(str(result))
    mask_candids = result[1][0]
    mask = mask_candids[0]
    for i in range(len(mask_candids)):
        if result[0][0][i][-1] >= confidence_threshold:
            mask = mask |  mask_candids[i]

    truthful = []
    for i in range(len(mask)):
        for j in (range(len(mask[0]))):
            if mask[i][j] == True:
                truthful.append((i,j))

    base = mmcv.image.rgb2gray(frame, keepdim=True)
    base = np.reshape(base, (v_in.height, v_in.width))
    new = np.stack((base,)*3, axis=-1)
    for e in truthful:
        new[e[0]][e[1]] = frame[e[0]][e[1]]

    print('============================')
    print(file)
    mmcv.imwrite(new, file)

mmcv.frames2video(temp_dir, dst_file, fps=v_in.fps, fourcc='mp4v')
