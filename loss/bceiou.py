from .SoftiouLoss import SoftIoULoss_func
from .bceLoss import BCE_loss
import torch.nn as nn
import torch
class Bce_IoU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,pred,target):
        bce_loss = 1.5*BCE_loss(pred,target)
        soft_IoU = 1*SoftIoULoss_func(pred,target)
        loss = bce_loss+soft_IoU
        return loss