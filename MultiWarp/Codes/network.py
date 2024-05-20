import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
from loguru import logger

import torchvision.transforms as T
resize_512 = T.Resize((512,512))

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2
def draw_mesh_on_warp(warp, f_local):

    warp = np.ascontiguousarray(warp)

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


# 对rigid_mesh应用全局单应矩阵变换
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2



# random augmentation
# it seems to do nothing to the performance
def data_aug(img1, img2):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img2_aug = img2 * random_brightness

    # Randomly shift color
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug  *= color_image

    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug  *= color_image

    # clip
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)

    return img1_aug, img2_aug

# for multi train.py / test.py
def build_model(net, input_tensors, is_training=True):
    batch_size, _, img_h, img_w = input_tensors[0].size()
    out_dict = {}  # 存储输出结果

    # network
    if is_training == True:
        # aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        H_motions, mesh_motions, tar_ids = net(input_tensors)
    else:
        H_motions, mesh_motions, tar_ids = net(input_tensors)

    out_dicts = []
    for i, tar in enumerate(tar_ids):
        H_motions[i] = H_motions[i].reshape(-1, 4, 2)
        mesh_motions[i] = mesh_motions[i].reshape(-1, grid_h+1, grid_w+1, 2)

        # initialize the source points bs x 4 x 2
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        # target points
        dst_p = src_p + H_motions[i]
        # solve homo using DLT
        H = torch_DLT.tensor_DLT(src_p, dst_p)

        M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                        [0., img_h / 2.0, img_h / 2.0],
                        [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        mask = torch.ones_like(input_tensors[tar])
        if torch.cuda.is_available():
            mask = mask.cuda()
        output_H = torch_homo_transform.transformer(torch.cat((input_tensors[tar], mask), 1), H_mat, (img_h, img_w))

        H_inv_mat = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H)), M_tile)
        output_H_inv = torch_homo_transform.transformer(torch.cat((input_tensors[1], mask), 1), H_inv_mat, (img_h, img_w))

        rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
        ini_mesh = H2Mesh(H, rigid_mesh)
        mesh = ini_mesh + mesh_motions[i]

        norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
        norm_mesh = get_norm_mesh(mesh, img_h, img_w)

        output_tps = torch_tps_transform.transformer(torch.cat((input_tensors[tar], mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
        warp_mesh = output_tps[:,0:3,...]
        warp_mesh_mask = output_tps[:,3:6,...]

        # calculate the overlapping regions to apply shape-preserving constraints
        overlap = torch_tps_transform.transformer(warp_mesh_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
        overlap = overlap.permute(0, 2, 3, 1).unfold(1, int(img_h/grid_h), int(img_h/grid_h)).unfold(2, int(img_w/grid_w), int(img_w/grid_w))
        overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
        overlap_one = torch.ones_like(overlap)
        overlap_zero = torch.zeros_like(overlap)
        overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)

        out_dict.update(success=True, output_H=output_H, output_H_inv = output_H_inv, warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, mesh1 = rigid_mesh, mesh2 = mesh, overlap = overlap)
        out_dicts.append(out_dict)

    return out_dicts, tar_ids


# for test_other.py
def build_new_ft_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    H_motion, mesh_motion = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    #H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)

    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    #mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)


    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]


    out_dict = {}
    out_dict.update(warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, rigid_mesh = rigid_mesh, mesh = mesh)


    return out_dict

# for train_ft.py
def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh):
    batch_size, _, img_h, img_w = input1_tensor.size()

    rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    mesh = torch.stack([mesh[...,0]*img_w/512, mesh[...,1]*img_h/512], 3)

    ######################################
    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    print(out_width)
    print(out_height)

    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = (input1_tensor+1)*127.5

    mask1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = 255

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()

    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

    stitch_tps_out = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask], 1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
    warp2 = stitch_tps_out[:,0:3,:,:]*127.5
    mask2 = stitch_tps_out[:,3:6,:,:]*255

    stitched = warp1*(warp1/(warp1+warp2+1e-6)) + warp2*(warp2/(warp1+warp2+1e-6))

    stitched_mesh = draw_mesh_on_warp(stitched[0].cpu().detach().numpy().transpose(1,2,0), mesh_trans[0].cpu().detach().numpy())

    out_dict = {}
    out_dict.update(warp1 = warp1, mask1 = mask1, warp2 = warp2, mask2 = mask2, stitched = stitched, stitched_mesh = stitched_mesh)

    return out_dict


# for multi test_output.py(net, input_tensors):
def build_output_model(net, input_tensors):
    batch_size, _, img_h, img_w = input_tensors[0].size()   
    out_dict = {}  # 存储输出结果

    H_motions, mesh_motions, tar_ids = net(input_tensors)
 
    # 确定拼接结果的画布尺寸
    width_max = torch.tensor(img_w).cuda()
    width_min = torch.tensor(0).cuda()
    height_max = torch.tensor(img_h).cuda()
    height_min = torch.tensor(0).cuda()

    meshes = []
    for i, tar in enumerate(tar_ids):
        H_motions[i] = H_motions[i].reshape(-1, 4, 2)  # 由1*8的二维向量变为1*4*2的三维向量
        H_motions[i] = torch.stack([H_motions[i][...,0]*img_w/512, H_motions[i][...,1]*img_h/512], 2)  # H_motion[...,0]为第0列，H_motion[...,1]为第1列，单独取出来缩放后再堆叠出去
        mesh_motions[i] = mesh_motions[i].reshape(-1, grid_h+1, grid_w+1, 2)  # 由1*338的二维向量变为1*13*13*2的四维向量
        mesh_motions[i] = torch.stack([mesh_motions[i][...,0]*img_w/512, mesh_motions[i][...,1]*img_h/512], 3)

        # initialize the source points bs x 4 x 2
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        # target points
        dst_p = src_p + H_motions[i]
        # solve homo using DLT
        H = torch_DLT.tensor_DLT(src_p, dst_p)

        rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)  # 划分网格，1*13*13*2的四维向量
        ini_mesh = H2Mesh(H, rigid_mesh)  # 应用全局单应变换
        mesh = ini_mesh + mesh_motions[i]  # 叠加每个网格顶点的运动
        width_max = torch.maximum(torch.max(mesh[...,0]), width_max)
        width_min = torch.minimum(torch.min(mesh[...,0]), width_min)
        height_max = torch.maximum(torch.max(mesh[...,1]), height_max)
        height_min = torch.minimum(torch.min(mesh[...,1]), height_min)
        meshes.append(mesh)

    out_width = width_max - width_min
    out_height = height_max - height_min
    print(f"out_width: {out_width}, out_height: {out_height}")
    if (out_height > 2000 or out_width > 100000):  # 拼接失败处理
        out_dict.update(success=False)
        return out_dict

    # 准备输出结果
    final_warps = [None] * len(input_tensors)
    final_warp_masks = [None] * len(input_tensors)
    final_meshes = [None] * len(input_tensors)
    
    # 处理公共参考图像，不进行变形，只应用缩放和平移矩阵
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)  # 划分网格，1*13*13*2的四维向量
    M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                      [0., out_height / 2.0, out_height / 2.0],
                      [0., 0., 1.]])
    N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
        N_tensor = N_tensor.cuda()
    N_tensor_inv = torch.inverse(N_tensor)

    I_ = torch.tensor([[1., 0., width_min],
                      [0., 1., height_min],
                      [0., 0., 1.]])#.unsqueeze(0)
    mask = torch.ones_like(input_tensors[1])
    if torch.cuda.is_available():
        I_ = I_.cuda()
        mask = mask.cuda()
    I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)

    ref_output = torch_homo_transform.transformer(torch.cat((input_tensors[1]+1, mask), 1), I_mat, (out_height.int(), out_width.int()))
    final_warps[1]=ref_output[:, 0:3, ...]-1
    final_warp_masks[1] = ref_output[:, 3:6, ...]
    final_meshes[1] = rigid_mesh

    torch.cuda.empty_cache()

    # 处理目标图像，应用网格和TPS变形
    for i, tar in enumerate(tar_ids):
        mesh_trans = torch.stack([meshes[i][...,0]-width_min, meshes[i][...,1]-height_min], 3)
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
        norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
        tps_output = torch_tps_transform.transformer(torch.cat([input_tensors[tar]+1, mask],1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
        final_warps[tar]=tps_output[:, 0:3, ...]-1
        final_warp_masks[tar] = tps_output[:, 3:6, ...]
        final_meshes[tar] = mesh_trans

    # 输出结果
    out_dict.update(success=True, final_warps=final_warps, final_warp_masks=final_warp_masks, final_meshes=final_meshes)
    return out_dict



# define and forward
class MultiWarpNetwork(nn.Module):

    # 网络由两部分组成：regressNet1和regressNet2，每个部分又分为卷积部分(part1)和全连接部分(part2)
    def __init__(self, input_img_num: int):
        super(MultiWarpNetwork, self).__init__()
        self.input_img_num = input_img_num

        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )


        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)

        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 所有卷积层使用Kaiming Normal初始化方法
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):  # 批量归一化层(nn.BatchNorm2d)设置其权重为1，偏置为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        ssl._create_default_https_context = ssl._create_unverified_context  # 解决SSL证书验证问题的临时方案，确保下载预训练模型时不会因SSL验证失败而中断
        resnet50_model = models.resnet.resnet50(pretrained=True)  # 加载预训练的ResNet50模型，作为特征提取的基础
        if torch.cuda.is_available():
            resnet50_model = resnet50_model.cuda()
        
        # 从ResNet50模型中获取特定阶段的特征图提取器
        self.feature_extractor_stage1, self.feature_extractor_stage2 = self.get_res50_FeatureMap(resnet50_model)
        #-----------------------------------------

    # 从预训练的ResNet50模型中提取特定的特征层，用于构建两个独立的特征提取器
    def get_res50_FeatureMap(self, resnet50_model):

        layers_list = []

        layers_list.append(resnet50_model.conv1)
        layers_list.append(resnet50_model.bn1)
        layers_list.append(resnet50_model.relu)
        layers_list.append(resnet50_model.maxpool)
        layers_list.append(resnet50_model.layer1)
        layers_list.append(resnet50_model.layer2)

        # 较低级特征提取
        feature_extractor_stage1 = nn.Sequential(*layers_list)

        # 较高级特征提取
        feature_extractor_stage2 = nn.Sequential(resnet50_model.layer3)

        #layers_list.append(resnet50_model.layer3)

        return feature_extractor_stage1, feature_extractor_stage2

    # multi forward
    def forward(self, input_tensors):
        batch_size, _, img_h, img_w = input_tensors[0].size()

        # 提取所有输入图像的特征
        features_64, features_32 = [], []
        for i in range(self.input_img_num):
            features_64.append(self.feature_extractor_stage1(input_tensors[i]))
            features_32.append(self.feature_extractor_stage2(features_64[i]))
        
        out_H_motions, out_mesh_motions, tar_ids = [], [], []
        for i in range(self.input_img_num):
            ref, tar = 1, i  # 选择1为参考图像，i为目标图像
            if ref == tar:
                continue
            tar_ids.append(tar)

            ######### stage 1
            correlation_32 = self.CCL(features_32[ref], features_32[tar])
            temp_1 = self.regressNet1_part1(correlation_32)
            temp_1 = temp_1.view(temp_1.size()[0], -1)
            offset_1 = self.regressNet1_part2(temp_1)  # offset_1为输出的H_motion
            H_motion_1 = offset_1.reshape(-1, 4, 2)  # 三维张量，形状1*4*2
            out_H_motions.append(offset_1)

            src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])  # 完整图像左上右上左下右下四个顶点
            if torch.cuda.is_available():
                src_p = src_p.cuda()
            src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)  # 转换成三维，形状1*4*2
            dst_p = src_p + H_motion_1  # 四个图像顶点的全局运动
            H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)  # 通过DLT求解H矩阵，3维张量，形状1*3*3，最后一个元素为1

            # 仿射变换矩阵M，将图像的宽和高分别缩小到原来的1/8，并将图像中心移动到原点
            M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                        [0., img_h/8 / 2.0, img_h/8 / 2.0],
                        [0., 0., 1.]])
            if torch.cuda.is_available():
                M_tensor = M_tensor.cuda()

            # 首先，通过M_tile_inv将原始坐标变换到计算H时所使用的统一缩放和平移坐标系。
            # 然后，应用单应性变换H，实现图像间的透视变换。
            # 最后，通过M_tile将变换结果转换回原始图像坐标系，使得变换后的图像能够正确地对齐到原始图像的坐标系统中
            M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            M_tensor_inv = torch.inverse(M_tensor)
            M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
            H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

            warp_feature_64_tar = torch_homo_transform.transformer(features_64[tar], H_mat, (int(img_h/8), int(img_w/8)))

            ######### stage 2
            correlation_64 = self.CCL(features_64[ref], warp_feature_64_tar)
            temp_2 = self.regressNet2_part1(correlation_64)
            temp_2 = temp_2.view(temp_2.size()[0], -1)
            offset_2 = self.regressNet2_part2(temp_2)  # offset_为输出的Mesh_motion
            out_mesh_motions.append(offset_2)

        return out_H_motions, out_mesh_motions, tar_ids


    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)
        #print(match_vol .size())

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        #print(flow.size())

        return feature_flow
