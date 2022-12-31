# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/29 22:51
@File: process_pth.py
@Desc: 
"""
import os.path

import torch

model_dir = '../output'
model_name = 'ckpt_epoch_49.pth'

ckpt = torch.load(os.path.join(model_dir, model_name))
new_ckpt = {}
for key, value in ckpt.items():
    if key in ['model', 'config']:
        new_ckpt[key] = value
torch.save(new_ckpt, os.path.join(model_dir, 'swin_t.pth'))