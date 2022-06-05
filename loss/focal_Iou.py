### 等待完成的
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SoftiouLoss import *
from .focalloss import *
class FocalSoftIoU(nn.Module):
    def __init__(self,alpha_1 = 1.0,alpha_2 = 1.0):
        super(FocalSoftIoU,self).__init__()
        self.softloss = alpha_1
        self.focaloss = alpha_2
        
    def forward(self,pred,target):
        default_sigmoid = False
        soft_iou_loss = SoftIoULoss_func(pred,target)
        focal_loss = py_sigmoid_focal_loss(pred,target)
        loss = self.softloss*soft_iou_loss+self.focaloss*focal_loss
        return loss




        