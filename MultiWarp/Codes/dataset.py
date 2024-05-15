from torch.utils.data import Dataset
import re
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random
from loguru import logger


class MultiWarpTrainDataset(Dataset):
    def __init__(self, data_path: str, input_img_num: int):

        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.input_img_num = input_img_num
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        pattern = r'^\d{2}$'  # 匹配正好两位的数字，包括前导零
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if bool(re.match(pattern, data_name)):
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):

        input_tensors = []
        for i in range(self.input_img_num):
            input_img = cv2.imread(self.datas['{:02d}'.format(index)]['image'][i])
            input_img = cv2.resize(input_img, (self.width, self.height))
            input_img = input_img.astype(dtype=np.float32)
            input_img = (input_img / 127.5) - 1.0
            input_img = np.transpose(input_img, [2, 0, 1])
            input_tensor = torch.tensor(input_img)
            input_tensors.append(input_tensor)

        # if_exchange = random.randint(0,1)
        # if if_exchange == 0:
        #     #print(if_exchange)
        #     return (input1_tensor, input2_tensor)
        # else:
        #     #print(if_exchange)
        #     return (input2_tensor, input1_tensor)

        return input_tensors

    def __len__(self):

        return len(self.datas.keys())

class MultiWarpTestDataset(Dataset):
    def __init__(self, data_path, use_resize=False):

        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.use_resize = use_resize
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))  # test_path 路径下的所有文件夹
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        logger.info(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        if (self.use_resize):
            input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        if (self.use_resize):
            input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])



