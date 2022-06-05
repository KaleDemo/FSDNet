from black import should_split_line
from model.UNet_Dyhead import UNetDyhead,UNetDyheadv2,UNetBasic
from model.UNet3plus import UNet3Plus_DeepSup
from ptflops import get_model_complexity_info
import torch
import os
import os.path as osp
import torch.nn as nn
import seaborn as sns
import cv2 as cv2
from thop import profile
""" 
    此python 文件实现的主要功能：
     1. 提供模型推理接口
     2. 提供FlOPS 计算接口
     3. 提供模型参数接口
     4. 提供Heatmap接口，大致就是这样
"""
class Inference(object):
    def __init__(self,args,save_path = None):
        self.model = args.model_name #  初始化对应的权重
        self.model.load_state_dict(torch.load(args.best_weight_path)) 
        self.save_path = save_path
    def operate(self,img_path,save_flag = True,show_flag = False):
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img,(320,320)).cuda()
        #  转换成对应成Tensor 
        img = img_resize.permute(2,0,1).unsqueeze(0)
        # (N,C,H,W) -> (H,W,C)
        predict = self.model(img).squeeze(0).permute(1,2,0)
        predict = predict.detach().cpu().numpy()
        # 转换成对应的张量的表示
        _, name = osp.split(img_path)
        if(save_flag):
            if(not self.save_path):
                cv2.imwrite(osp.join(self.save_path,name),predict)
            else:
                cv2.imwrite(osp.join(os.getcwd(),name),predict)
        if(show_flag):
            cv2.imshow('Inference',predict)
    def heatmap(self,img_path,sns_path):
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img,(320,320)).cuda()
        #  转换成对应成Tensor 
        img = img_resize.permute(2,0,1).unsqueeze(0)
        # (N,C,H,W) -> (H,W,C)
        predict = self.model(img).squeeze(0).permute(1,2,0)
        predict = predict.detach().cpu().numpy()
        # 转换成对应的张量的表示
        heatmap = sns.heatmap(predict[:,:,0],cmap="YlGnBu")
        _, name = osp.split(img_path)
        sns_fig = heatmap.get_figure()
        sns_fig.savefig(osp.join(sns_path,name),dpi=400)

   




        


