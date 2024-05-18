# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_output_model, MultiWarpNetwork
from dataset import MultiWarpTestDataset
import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
import setproctitle
from loguru import logger
from datetime import datetime

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  # UDIS2/MultiWarp 文件夹
DATASET_ROOT = "/home/B_UserData/dongzhipeng/Datasets"
MODEL_DIR = os.path.join(PROJ_ROOT, 'model/')


def draw_mesh_on_warp(warp, f_local):

    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return




def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define dataset
    logger.info('<==================== Loading data ===================>')
    test_path = os.path.join(DATASET_ROOT, args.test_path)
    test_data = MultiWarpTestDataset(data_path=test_path, input_img_num=args.input_img_num, use_resize=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)  # num_workers: the number of cpus

    # define the network
    logger.info('<==================== Defining network ===================>')
    net = MultiWarpNetwork(input_img_num=args.input_img_num)
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    # 加载预训练模型
    logger.info('<==================== Loading ckpt ===================>')
    model_path = os.path.join(MODEL_DIR, args.model)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        logger.info('load model from {}!'.format(model_path))
    else:
        logger.info('training from stratch!')


    logger.info('<==================== start testing ===================>')

    for i, batch_value in tqdm(enumerate(test_loader)):
        input_tensors = []
        for img_idx in range(args.input_img_num):
            input_tensor = batch_value[img_idx].float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            input_tensors.append(input_tensor)

        with torch.no_grad():
            batch_out = build_output_model(net, input_tensors)

        final_warps = batch_out['final_warps']
        final_warp_masks = batch_out['final_warp_masks']
        final_meshes = batch_out['final_meshes']

        # 创建保存结果文件夹
        batch_path = test_data.get_path(i)
        path_ave_fusion = os.path.join(batch_path, 'ave_fusion/')
        os.makedirs(path_ave_fusion, exist_ok=True)
        path_warp = os.path.join(batch_path, 'warp/')
        os.makedirs(path_warp, exist_ok=True)
        path_mask = os.path.join(batch_path, 'mask/')
        os.makedirs(path_mask, exist_ok=True)

        for j in range(args.input_img_num):
            final_warps[j] = ((final_warps[j][0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            final_warp_masks[j] = final_warp_masks[j][0].cpu().detach().numpy().transpose(1,2,0)
            final_meshes[j] = final_meshes[j][0].cpu().detach().numpy()

            cv2.imwrite(os.path.join(path_warp, str(j) + ".jpg"), final_warps[j])
            cv2.imwrite(os.path.join(path_mask, str(j) + ".jpg"), final_warp_masks[j]*255)

        # 平均融合 ave_fusion = final_warp1 * (final_warp1/ (final_warp1+final_warp2+1e-6)) + final_warp2 * (final_warp2/ (final_warp1+final_warp2+1e-6))
        den = np.zeros_like(final_warps[0]) + 1e-6
        for j in range(args.input_img_num):
            den += final_warps[j]
        ave_fusion = np.zeros_like(final_warps[0])
        for j in range(args.input_img_num):
            ave_fusion += final_warps[j] * (final_warps[j] / den)
        cv2.imwrite(os.path.join(path_ave_fusion, batch_path.split('/')[-1]+"_"+str(args.input_img_num)+".jpg"), ave_fusion)

        torch.cuda.empty_cache()

    logger.info('<==================== end testing ===================>')


if __name__=="__main__":

    setproctitle.setproctitle("dongzhipeng_test")

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--input_img_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='MiniTank1/testing/')
    parser.add_argument('--model', type=str, default='MiniTank1_20240518_201535/epoch200.pth')  # MODEL_DIR 下的模型文件

    args = parser.parse_args()
    print(args)
    
    test(args)
