import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dyhead import DyHead,DyDCNv2
from model.dyhead import DyHeadBlock
from model.UNet3plus import UNet3Plus,UNet3Plus2
from .layers import *
class UNetDyhead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = UNet3Plus()
        self.Dyhead = DyHead(in_channels=80,out_channels=80)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(240,1,kernel_size=1, stride=1)
    def forward(self,x):
        hd1,hd2,hd3 =  self.backbone(x)
        hd1,hd2,hd3 =  self.Dyhead([hd1,hd2,hd3])
        hd2 = self.up(hd2)
        hd3 = self.up_4(hd3)
        final =  torch.cat([hd1,hd2,hd3],dim=1)
        res = self.final(final)
        ## 分别是对应的下采样的demo
        return res
class UNetBasic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = UNet3Plus2()
        self.Dept_conv_low = nn.Conv2d(256,80,kernel_size=1, stride=1)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,   mode='bilinear', align_corners=True)
        self.ResCBM = self._make_layer(block=Res_CBAM_block,input_channels=400,output_channels=16,num_blocks=2)
        # 到这里将所有的转换成上述的算法
        self.Depth_conv = nn.Conv2d(16,1,kernel_size=1, stride=1)
    def forward(self,x):
        hd1,hd2,hd3,hd4,hd5 = self.backbone(x)
        hd5 = self.Dept_conv_low(hd5)
        hd2 = self.up(hd2)
        hd3 = self.up_4(hd3)
        hd4 = self.up_8(hd4)
        hd5 = self.up_16(hd5)
        final = torch.cat([hd1,hd2,hd3,hd4,hd5],dim=1)
        final = self.ResCBM(final)
        final = self.Depth_conv(final)
        return final
    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

class UNetDyheadv2(nn.Module):
    """
        慎用，因为俺也不知道会发生什么,
    """
    def __init__(self,deep_supervision = False) -> None:
        super().__init__()
        self.backbone = UNet3Plus2()
        self.Dyhead_up = DyHead(in_channels=80,out_channels=80)
        self.Dyhead_down = DyHead(in_channels=80,out_channels=80)
        self.Dept_conv_low = nn.Conv2d(256,80,kernel_size=1, stride=1)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,   mode='bilinear', align_corners=True)
        self.ResCBM = self._make_layer(block=Res_CBAM_block,input_channels=400,output_channels=16,num_blocks=2)
        self.deep_supervision = deep_supervision
        # 到这里将所有的转换成上述的算法
        self.Depth_conv = nn.Conv2d(16,1,kernel_size=1, stride=1)
        if deep_supervision:
            self.conv2 = nn.Conv2d(80,1,kernel_size=1, stride=1)
            self.conv4 = nn.Conv2d(80,1,kernel_size=1, stride=1)
            self.conv8 = nn.Conv2d(80,1,kernel_size=1, stride=1)
            self.conv16 = nn.Conv2d(80,1,kernel_size=1, stride=1)
    def forward(self,x):
        hd1,hd2,hd3,hd4,hd5 = self.backbone(x)
        hd5 = self.Dept_conv_low(hd5)
        hd3,hd4,hd5 =  self.Dyhead_down([hd3,hd4,hd5])
        hd1,hd2,hd3 =  self.Dyhead_up([hd1,hd2,hd3])
        hd2 = self.up(hd2)
        hd3 = self.up_4(hd3)
        hd4 = self.up_8(hd4)
        hd5 = self.up_16(hd5)
        if(self.deep_supervision):
            final = torch.cat([hd1,hd2,hd3,hd4,hd5],dim=1)
            final = self.ResCBM(final)
            final = self.Depth_conv(final)
            hd2 = self.conv2(hd2)
            hd3 = self.conv4(hd3)
            hd4 = self.conv8(hd4)
            hd5 = self.conv16(hd5)
            return [final,hd2,hd3,hd4,hd5]
        else:
            final = torch.cat([hd1,hd2,hd3,hd4,hd5],dim=1)
            final = self.ResCBM(final)
            final = self.Depth_conv(final)
            return final
    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)


        # 三个FPN的层次，调用那个人的方式，然后再进行通道注意力机制的融合