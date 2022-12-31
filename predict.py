# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/11/18 20:29
@File: predict.py
@Desc: 
"""
import json
import os
import time
import torch
from PIL import Image
from models import build_model
from data.build import build_transform
import logging

logger = logging.getLogger(__name__)


class Predict:
    def __init__(self, model_dir, model_name, device='cpu'):
        self.use_gpu = False if 'cpu' in device else True
        device = torch.device(device)
        # 加载模型
        ckpt = torch.load(os.path.join(model_dir, model_name), map_location='cpu')
        # 配置
        config = ckpt['config']
        # 模型
        model = build_model(config, train=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        if self.use_gpu:
            model.cuda()
        # 类别
        with open(os.path.join(model_dir, 'class_to_idx.json'), 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        self.model = model
        self.transforms = build_transform(False, config)
        print("Init Model Finished!", model_name)

    def __call__(self, image):
        tensor = self.transforms(image)
        if self.use_gpu:
            tensor = tensor.cuda()
        tensor = tensor.unsqueeze_(0)
        output = self.model(tensor)
        output = torch.max(output, 1)[1].cpu()
        cls_name = self.idx_to_class[int(output[0])]
        return cls_name


if __name__ == '__main__':
    image_path = '/teams/2221AI13_1666364726/garbage/val/其他垃圾_一次性杯子/1014.jpg'
    image = Image.open(image_path)
    image = image.convert('RGB')

    model_dir = '/mnt/trained_model/swin_base_patch4_window7_224/202211180116'
    model_name = 'ckpt_epoch_37.pth'
    predict = Predict(model_dir, model_name)
    start = time.time()
    pred_name = predict(image)
    end = time.time()
    print('time: {}, pred: {}'.format(end - start, pred_name))