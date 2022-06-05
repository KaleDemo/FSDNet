import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from MoCoPnet import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr2, ssim
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import numpy as np
parser = argparse.ArgumentParser(description="PyTorch MoCoPnet")
parser.add_argument("--datasets", type=str, default=['SAITD','Hui'], help="Test datasets")
parser.add_argument("--ckpt", default='./log/MoCoPnet.pth.tar', type=str, help="checkpoint path")
parser.add_argument("--scale_factor", type=int, default=4, help="scale factor")
parser.add_argument("--device", type=int, default=0, help="GPU id")

global opt
opt = parser.parse_args()
class Demo(nn.Module):
    def __init__(self,opt=opt):
        super().__init__()
        self.net = Net(4)
        self.device = torch.device('cuda')
        ckpt = torch.load(opt.ckpt, map_location='cuda:0')
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.to(self.device)
    def __call__(self,input_img,save_path,save_flag = True):
        """
            仅仅支持固定倍率的计算的方式
        """
        LR = []
        for i in range(7):
            img_LR = Image.open(input_img)
            img_LR = np.array(img_LR,dtype = np.float32)/255.0
            img_LR = img_LR[:,:,0]
            img_LR = np.array(img_LR,dtype = np.float32)[np.newaxis,:]
            LR.append(img_LR)
        LR = np.stack(LR,0)
        LR = torch.from_numpy(np.ascontiguousarray(LR.copy()))
        LR = Variable(LR).cuda()#转换成对应的张量计算
        LR = LR.unsqueeze(0)
        print(LR.shape)
        SR = self.net(LR)
        SR = torch.clamp(SR,0,1)
        print(SR.shape)
        SR_img = transforms.ToPILImage()(SR[0,:,:,:].cpu())
        
        if(save_flag):
            SR_img.save(save_path)
        return SR_img   




def main():
    demo_parse = Demo(opt=opt)
    image = demo_parse('/home/wa/MPANet/256_sirst/image_split/Misc_7__2__234___104.png','/home/wa/QAQ.png')







if __name__ == '__main__':
    main()

