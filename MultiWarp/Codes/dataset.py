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
                img_lists = glob.glob(os.path.join(data, '*.jpg'))
                if len(img_lists) < self.input_img_num:
                    continue
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = img_lists
                self.datas[data_name]['image'].sort()
        self.data_keys = list(self.datas.keys())
        logger.info(self.data_keys)

    def __getitem__(self, index):

        input_tensors = []
        for i in range(self.input_img_num):
            input_img = cv2.imread(self.datas[self.data_keys[index]]['image'][i])
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
    
    def get_path(self, index):

        return self.datas[self.data_keys[index]]['path']

class MultiWarpTestDataset(Dataset):
    def __init__(self, data_path: str, input_img_num: int, use_resize=True):

        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.input_img_num = input_img_num
        self.use_resize = use_resize
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))  # test_path 路径下的所有文件夹
        pattern = r'^\d{2}$'  # 匹配正好两位的数字，包括前导零
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if bool(re.match(pattern, data_name)):
                img_lists = glob.glob(os.path.join(data, '*.jpg'))
                if len(img_lists) < self.input_img_num:
                    continue
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = img_lists
                self.datas[data_name]['image'].sort()
        self.data_keys = list(self.datas.keys())
        logger.info(self.data_keys)

    def __getitem__(self, index):

        input_tensors = []
        for i in range(self.input_img_num):
            input_img = cv2.imread(self.datas[self.data_keys[index]]['image'][i])
            if (self.use_resize):
                input_img = cv2.resize(input_img, (self.width, self.height))
            input_img = input_img.astype(dtype=np.float32)
            input_img = (input_img / 127.5) - 1.0
            input_img = np.transpose(input_img, [2, 0, 1])
            input_tensor = torch.tensor(input_img)
            input_tensors.append(input_tensor)

        return input_tensors

    def __len__(self):

        return len(self.datas.keys())
    
    def get_path(self, index):
        
        return self.datas[self.data_keys[index]]['path']



