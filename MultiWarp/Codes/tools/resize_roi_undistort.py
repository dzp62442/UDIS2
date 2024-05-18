"""
将多路环视相机采集的图像序列进行预处理，包括：调整图像分辨率、截取 ROI、去畸变等。
"""
import numpy as np
import cv2
import os
from tqdm import tqdm

#! 设置参数
DATASET_ROOT = "/home/B_UserData/dongzhipeng/Datasets/MiniTank1"
ORIGIN_DIR = os.path.join(DATASET_ROOT, "origin")
PROCESSED_DIR = os.path.join(DATASET_ROOT, "1280720_undistorted")
OUT_W, OUT_H = 1280, 1024
ROI_W, ROI_H = 1280, 720
EXPS = ['ex_0']
CAM_NUM = 6
SUFFIX = ".jpg"
K = np.array([[794.7633, 0.0, 635.6471], [0.0, 794.8898, 369.4420], [0.0, 0.0, 1.0]])
D = np.array([-0.3489, 0.1329, 0.0, 0.0])


def resize_image(image, target_resolution):
    height, width = image.shape[:2]
    if (height, width) != target_resolution:
        image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)
    return image

def crop_to_roi(image, roi_w, roi_h):
    height, width = image.shape[:2]
    if height != roi_h:
        y = (height - roi_h) // 2
        image = image[y:y+roi_h, :]
        return image

# 生成undistort映射矩阵
def generate_remap():
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (ROI_W, ROI_H), 1, (ROI_W, ROI_H))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_camera_matrix, (ROI_W, ROI_H), 5)
    return mapx, mapy, roi

# 直接去畸变，分辨率不变，但左右两侧丢失较多
def undistort_image(image):
    undistorted_image = cv2.undistort(image, K, D)
    return undistorted_image

# 利用remap去畸变，分辨率降低且宽高比变大，但左右两侧丢失较少
def undistort_image_remap(image, mapx, mapy, roi):
    undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistorted_image

def main():
    mapx, mapy, roi = generate_remap()
    for exp in EXPS:
        for cam_idx in range(CAM_NUM):
            read_path = os.path.join(ORIGIN_DIR, exp, f"video{cam_idx}")
            save_path = os.path.join(PROCESSED_DIR, exp, f"video{cam_idx}")
            os.makedirs(save_path, exist_ok=True)
            print("Processing: ", read_path, " -> ", save_path)
            # 遍历文件夹中的图片
            files = sorted(os.listdir(read_path))
            for filename in tqdm(files):
                if filename.endswith(SUFFIX):
                    # 读取图片
                    image_path = os.path.join(read_path, filename)
                    image = cv2.imread(image_path)

                    # 判断分辨率是否一致，若不一致则进行resize
                    image = resize_image(image, (OUT_W, OUT_H))

                    # 判断resize后的分辨率和ROI是否一致，若不一致则从中央截取ROI
                    image = crop_to_roi(image, ROI_W, ROI_H)

                    # 根据给定的相机内参和畸变系数进行去畸变
                    image = undistort_image(image)
                    # image = undistort_image_remap(image, mapx, mapy, roi)

                    # 处理后的图片可以进一步操作，例如保存或显示
                    cv2.imwrite(os.path.join(save_path, filename), image)

if __name__ == "__main__":
    main()
