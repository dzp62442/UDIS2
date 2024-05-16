import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, MultiWarpNetwork
from dataset import MultiWarpTrainDataset
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss
import glob
from loguru import logger
import setproctitle
from datetime import datetime


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))  # UDIS2/MultiWarp 文件夹
DATASET_ROOT = "/home/B_UserData/dongzhipeng/Datasets"
MODEL_DIR = os.path.join(PROJ_ROOT, 'model/')
SUMMARY_DIR = os.path.join(PROJ_ROOT, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define dataset
    train_path = os.path.join(DATASET_ROOT, args.train_path)
    train_data = MultiWarpTrainDataset(data_path=train_path, input_img_num=args.input_img_num)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # TODO define the network
    net = MultiWarpNetwork(input_img_num=args.input_img_num)
    if torch.cuda.is_available():
        net = net.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # 加载预训练模型
    model_path = os.path.join(MODEL_DIR, args.model)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        logger.info('load model from {} with start_epoch {}!'.format(model_path, start_epoch))
    else:
        start_epoch = 0
        glob_iter = 0
        logger.info('training from stratch!')


    logger.info('<==================== start training ===================>')
    score_print_fre = 10

    for epoch in range(start_epoch, args.max_epoch):

        logger.info("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.

        logger.info('epoch {}, lr={:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            input_tensors = []
            for img_idx in range(args.input_img_num):
                input_tensor = batch_value[img_idx].float()
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                input_tensors.append(input_tensor)

            # forward, backward, update weights
            optimizer.zero_grad()
            out_dicts, tar_ids = build_model(net, input_tensors)

            total_loss = 0
            for j, tar in enumerate(tar_ids):
                # result
                output_H = out_dicts[j]['output_H']
                output_H_inv = out_dicts[j]['output_H_inv']
                warp_mesh = out_dicts[j]['warp_mesh']
                warp_mesh_mask = out_dicts[j]['warp_mesh_mask']
                mesh1 = out_dicts[j]['mesh1']
                mesh2 = out_dicts[j]['mesh2']
                overlap = out_dicts[j]['overlap']

                # calculate loss for overlapping regions
                overlap_loss = cal_lp_loss(input_tensors[1], input_tensors[tar], output_H, output_H_inv, warp_mesh, warp_mesh_mask)
                # calculate loss for non-overlapping regions
                nonoverlap_loss = 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)

                total_loss += overlap_loss + nonoverlap_loss
            total_loss.backward()

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            # overlap_loss_sigma += overlap_loss.item()
            # nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_sigma += total_loss.item()

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                # average_overlap_loss = overlap_loss_sigma/ score_print_fre
                # average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.

                # print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                #                           average_loss, average_overlap_loss, average_nonoverlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, optimizer.state_dict()['param_groups'][0]['lr']))

                
                # visualization
                writer.add_image("input0", (input_tensors[0][0]+1.)/2., glob_iter)
                writer.add_image("input1", (input_tensors[1][0]+1.)/2., glob_iter)
                writer.add_image("input2", (input_tensors[2][0]+1.)/2., glob_iter)
                writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                # writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                # writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)

            glob_iter += 1

        scheduler.step()

        # save model
        if (epoch == start_epoch):  # 创建模型保存文件夹
            dataset_name = args.train_path.split('/')[0]
            now = datetime.now()
            model_save_dir = os.path.join(MODEL_DIR, dataset_name+'_'+now.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(model_save_dir, exist_ok=True)
        if ((epoch+1) % 1 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '.pth'
            model_save_path = os.path.join(model_save_dir, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
    
    logger.info('<==================== end training ===================>')


if __name__=="__main__":

    logger.info('<==================== setting arguments ===================>')

    setproctitle.setproctitle("dongzhipeng_train")
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--input_img_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_path', type=str, default='M-UDIS-D/training/')  # DATASET_ROOT 下的训练数据路径
    parser.add_argument('--model', type=str, default='warp.pth')  # MODEL_DIR 下的模型文件

    args = parser.parse_args()
    print(args)

    train(args)


