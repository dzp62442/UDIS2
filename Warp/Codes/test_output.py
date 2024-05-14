# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_output_model, WarpNetwork
from dataset import *
import os
import cv2
from tqdm import tqdm
import setproctitle
from loguru import logger

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

PROJ_ROOT = "/home/dongzhipeng/Projects/UDIS2"
DATASET_ROOT = "/home/B_UserData/dongzhipeng/Datasets"
MODEL_DIR = os.path.join(PROJ_ROOT, 'Warp/model/')


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
    
    # dataset
    logger.info('<==================== Loading data ===================>')
    test_data = WarpTestDataset(data_path=args.test_path, use_resize=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)  # num_workers: the number of cpus

    # define the network
    logger.info('<==================== Defining network ===================>')
    net = WarpNetwork()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    #load the existing models if it exists
    logger.info('<==================== Loading ckpt ===================>')
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        logger.info('load model from {}!'.format(model_path))
    else:
        logger.warning('No checkpoint found!')


    logger.info('<==================== start testing ===================>')
    # 创建保存结果文件夹
    path_ave_fusion = args.test_path + 'ave_fusion/'
    os.makedirs(path_ave_fusion, exist_ok=True)
    path_warp1 = args.test_path + 'warp1/'
    os.makedirs(path_warp1, exist_ok=True)
    path_warp2 = args.test_path + 'warp2/'
    os.makedirs(path_warp2, exist_ok=True)
    path_mask1 = args.test_path + 'mask1/'
    os.makedirs(path_mask1, exist_ok=True)
    path_mask2 = args.test_path + 'mask2/'
    os.makedirs(path_mask2, exist_ok=True)

    for i, batch_value in tqdm(enumerate(test_loader)):
        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        with torch.no_grad():
            batch_out = build_output_model(net, inpu1_tesnor, inpu2_tesnor)

        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']
        final_mesh1 = batch_out['mesh1']
        final_mesh2 = batch_out['mesh2']

        final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_mesh1 = final_mesh1[0].cpu().detach().numpy()
        final_mesh2 = final_mesh2[0].cpu().detach().numpy()

        path = path_warp1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1)
        path = path_warp2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2)
        path = path_mask1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1_mask*255)
        path = path_mask2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2_mask*255)

        ave_fusion = final_warp1 * (final_warp1/ (final_warp1+final_warp2+1e-6)) + final_warp2 * (final_warp2/ (final_warp1+final_warp2+1e-6))
        path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, ave_fusion)

        # print('i = {}'.format( i+1))

        torch.cuda.empty_cache()

    print("##################end testing#######################")


if __name__=="__main__":

    setproctitle.setproctitle("dongzhipeng_test")

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default=os.path.join(DATASET_ROOT, 'MiniTank1/testing/'))

    args = parser.parse_args()
    print(args)
    test(args)
