# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/11/20 14:50 
@File : test.py 
@Desc : 
'''

import json
import os
import shutil
from tqdm import tqdm
import torch
import numpy as np
from sklearn import metrics
from models import build_model
from data import build_loader_test
from utils.io_utils import report_confusion_to_excel

target_names_4 = ['可回收物', '厨余垃圾', '有害垃圾', '其他垃圾']
name_cls_map_4 = {}
cls_name_map_4 = {}
for cls_idx, cls_name in enumerate(target_names_4):
    name_cls_map_4[cls_name] = cls_idx
    cls_name_map_4[cls_idx] = cls_name


@torch.no_grad()
def test(config, data_loader, model, target_names=None):
    idx_248_4 = {}      # 小类idx -> 大类idx
    for cls_idx, cls_name in enumerate(target_names):
        cls_name_4 = cls_name.split('_')[0]
        cls_idx_4 = name_cls_map_4[cls_name_4]
        idx_248_4[cls_idx] = cls_idx_4
    save_dir = os.path.join(config.OUTPUT, 'test_res')
    os.makedirs(save_dir, exist_ok=True)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for idx, (images, target, paths) in enumerate(tqdm(data_loader)):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        # measure accuracy and record loss
        loss = criterion(output, target)
        loss_total += loss
        target = target.data.cpu().numpy()
        output = torch.max(output, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, target)
        predict_all = np.append(predict_all, output)
        # 将识别错的图片放到 gt/pred 文件夹下
        # for gt, pred, path in zip(target, output, paths):
        #     if gt == pred: continue
        #     gt_name = target_names[int(gt)]
        #     pred_name = target_names[int(pred)]
        #     cur_save_dir = os.path.join(save_dir, gt_name, pred_name)
        #     os.makedirs(cur_save_dir, exist_ok=True)
        #     dst_img_path = os.path.join(cur_save_dir, os.path.split(path)[-1])
        #     shutil.copy(path, dst_img_path)
    # 小类评测
    acc, recall, f1, _ = metrics.precision_recall_fscore_support(labels_all, predict_all, average='macro', zero_division=0)
    report = metrics.classification_report(labels_all, predict_all, digits=4, target_names=target_names, output_dict=True)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    msg = '248类：Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Rec: {2:>6.2%}, Test F1: {3:>6.2%}'
    print(msg.format(loss_total/len(data_loader), acc, recall, f1))
    # 大类评测
    # 将小类映射为大类
    for cls_idx, cls_name in enumerate(target_names):
        labels_all[labels_all==cls_idx] = idx_248_4[cls_idx]
        predict_all[predict_all==cls_idx] = idx_248_4[cls_idx]
    acc, recall, f1, _ = metrics.precision_recall_fscore_support(labels_all, predict_all, average='macro',zero_division=0)
    report_4 = metrics.classification_report(labels_all, predict_all, digits=4, target_names=target_names_4, output_dict=True)
    confusion_4 = metrics.confusion_matrix(labels_all, predict_all)
    msg = '4类：Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Rec: {2:>6.2%}, Test F1: {3:>6.2%}'
    excel_save_path = os.path.join(config.OUTPUT, 'test_20221231.xlsx')
    # add_confusion_matrix(writer, confusion, num_classes=len(target_names), class_names=target_names, tag="Test Confusion Matrix", figsize=[10,8])
    report_confusion_to_excel(report, confusion, target_names, excel_save_path, sheet_names=['小类-report', '小类-confusion'])
    report_confusion_to_excel(report_4, confusion_4, target_names_4, excel_save_path, sheet_names=['大类-report', '大类-confusion'])


if __name__ == '__main__':
    model_dir = '/mnt/trained_model/swin_base_patch4_window7_224/202211180116'
    # model_dir = r'D:\北航\学习资料\人工智能原理与技术\大作业\训练记录\swin_base_patch4_window7_224'
    model_name = 'ckpt_epoch_49.pth'
    print('load model: ', os.path.join(model_dir, model_name))
    # 加载模型
    ckpt = torch.load(os.path.join(model_dir, model_name), map_location='cpu')
    # 配置
    config = ckpt['config']

    config.defrost()
    config.DATA.BATCH_SIZE = 128                # 修改 batch size
    # config.DATA.DATA_PATH = r'C:\Users\吴志超\Downloads\garbage'       # 修改数据路径
    config.freeze()
    # 模型
    model = build_model(config, train=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    # 类别
    with open(os.path.join(model_dir, 'class_to_idx.json'), 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    # 修改数据路径
    dataset_test, data_loader_test = build_loader_test(config, class_to_idx)
    test(config, data_loader_test, model, target_names=target_names)