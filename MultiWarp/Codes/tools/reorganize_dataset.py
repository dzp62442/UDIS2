"""
将按时间顺序组织的车辆环视数据集改组为按空间顺序组织
按时间顺序组织：每个文件夹里保存的是单个相机在不同时刻的数据
按空间顺序组织：每个文件夹里保存的是同一时刻下多个相机的数据
"""

import numpy as np
import cv2
import os
from tqdm import tqdm

#! 设置参数
DATASET_ROOT = "/home/B_UserData/dongzhipeng/Datasets/MiniTank1"
ORIGIN_DIR = os.path.join(DATASET_ROOT, "origin")
PROCESSED_DIR = os.path.join(DATASET_ROOT, "1280720_undistorted")
REORGANIZED_DIR = os.path.join(DATASET_ROOT, "training")
EXPS = ['ex_0', 'ex_1']
CAM_NUM = 6
READ_INTERVAL = 300
SUFFIX = ".jpg"


def main():
    imgs = [None] * CAM_NUM
    save_idx = -1
    
    for exp in EXPS:
        exp_dir = os.path.join(PROCESSED_DIR, exp)
        frames_per_cam = len(os.listdir(os.path.join(exp_dir, "video"+str(CAM_NUM-1))))
        print("Frames per camera: ", frames_per_cam)
        for i in range(0, frames_per_cam, READ_INTERVAL):
            save_idx += 1
            os.makedirs(os.path.join(REORGANIZED_DIR, str(save_idx)), exist_ok=True)
            for cam in range(CAM_NUM):
                img = cv2.imread(os.path.join(exp_dir, "video"+str(cam), str(i).zfill(6)+SUFFIX))
                cv2.imwrite(os.path.join(REORGANIZED_DIR, str(save_idx), str(cam)+SUFFIX), img)
            print(f"Save {exp} frame {i} to folder {save_idx}")


if __name__ == "__main__":
    main()