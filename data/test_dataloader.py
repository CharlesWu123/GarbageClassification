# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/11/20 15:16 
@File : test_dataloader.py 
@Desc : 
'''
import os
import os.path

import torch.utils.data as data
from PIL import Image


class TestDataset(data.Dataset):
    def __init__(self, root, classes_to_idx, transform=None):
        self.samples = []
        self.root = root
        self.classes_to_idx = classes_to_idx
        self.idx_to_classes = {v:k for k, v in classes_to_idx.items()}
        self.classes = [self.idx_to_classes[i] for i in range(len(classes_to_idx))]
        self.transform = transform
        self.make_dataset()

    def make_dataset(self):
        dir_name_list = os.listdir(self.root)
        for dir_name in dir_name_list:
            file_name_list = os.listdir(os.path.join(self.root, dir_name))
            for file_name in file_name_list:
                file_path = os.path.join(self.root, dir_name, file_name)
                self.samples.append((file_path, self.classes_to_idx[dir_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path, target = self.samples[item]
        sample = Image.open(path)
        sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path