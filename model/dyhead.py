import torch
import torch.nn.functional as F
from torch import nn
from .deform import ModulatedDeformConv
from .dyrelu import h_sigmoid, DYReLU
"""
    Dynamic Head 面向
"""
class DyDCNv2(nn.Module):
    def __init__(self,in_channels,out_channels,stride =1):
        super(DyDCNv2,self).__init__()
        self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x
class DyHeadBlock(nn.Module):
    def __init__(self,in_channels,out_channels,zero_init_offset = True,act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3
        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),h_sigmoid())
        self.task_attn_module = DYReLU(in_channels,out_channels)
        self.init_weights()
    def init_weights(self):
        pass
    def forward(self,x):
        outs = []
        for level in range(len(x)):#计算对应的FPN的尺度的大小
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            conv_args = dict(offset=offset, mask=mask)
            mid_feat = self.spatial_conv_mid(x[level],**conv_args)
            sum_feat = mid_feat*self.scale_attn_module(mid_feat)
            summed_levels = 0     
            if level > 0:
              #  print(level,'stride')
                low_feat = self.spatial_conv_low(x[level - 1], **conv_args)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
               # print(level,'upsample')
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], **conv_args),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))
        return outs
class DyHead(nn.Module):
     def __init__(self,in_channels,out_channels,num_blocks = 1,zero_init_offset=True,init_cfg=None):
            assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
            super().__init__()
            self.in_channels = in_channels 
            self.out_channels = out_channels
            self.num_blocks = num_blocks
            self.zero_init_offset = zero_init_offset
            dyhead_blocks = []
            for i in range(num_blocks):
                in_channels = self.in_channels if i == 0 else self.out_channels
                dyhead_blocks.append(
                    DyHeadBlock(
                        in_channels,
                        self.out_channels,
                        zero_init_offset=zero_init_offset))
            self.dyhead_blocks = nn.Sequential(*dyhead_blocks)
     def forward(self,inputs):
        assert isinstance(inputs, (tuple, list))
        outs = self.dyhead_blocks(inputs)
        return tuple(outs)

    
  
       