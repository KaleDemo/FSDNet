import os 
import os
import logging
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
import torch
from model.hrnet import hrnet18,hrnet32

def main():
    demo_input = torch.randn(1,3,256,256)
    model = hrnet18(pretrained=False)
    demo_output = model(demo_input)
    print(demo_output.shape)

if __name__ =='__main__':
    main()