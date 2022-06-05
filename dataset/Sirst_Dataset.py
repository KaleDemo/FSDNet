# 简单来说，这是面向
from lib2to3.pytree import Base
from random import random
from torch import rand
from torchvision  import transforms
from torch.utils.data import Dataset
import cv2 as cv2
import PIL.Image as Image
import numpy as np
import pandas as pd
import os
import albumentations as A
import os.path as osp
import random
"""
    准备更新集成Mixup的数据增强的方式
"""
#### 

class SirTrain(Dataset):
    def __init__(self,BaseDir):
        super().__init__()
        self.base_dir = BaseDir
        self.normalize = True
        self.image_path = osp.join(self.base_dir,'img')
        self.mask_path = osp.join(self.base_dir,'labelcol')
        self.image_list = os.listdir(self.image_path)
        self.transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
            ]
        )
        self.augtransform = A.Compose(
            [ 
                A.Resize(height = 320,width = 320,interpolation=cv2.INTER_CUBIC,p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
            ]
            )
    def __getitem__(self,idx):
        cur_img = self.image_list[idx]
        image = cv2.imread(osp.join(self.image_path,cur_img))
        mask  = cv2.imread(osp.join(self.mask_path,cur_img))
        augmentation = self.augtransform(image = image,mask = mask)
        image = augmentation['image']
        mask = augmentation['mask']
        image,mask = self.Mixup(img_org=image,mask_org=mask)
        if(self.normalize):
            # 这里做归一化的原因在这里，focal loss的原因
            image = image.astype(np.float32)/255.0
            mask = mask.astype(np.float32)/255.0
        image = self.transform(image)
        mask = self.transform(mask)
        return image,mask
    def __len__(self):
        return len(self.image_list)
    def Mixup(self,img_org,mask_org,p=0.3):
        p = p
        if (random.random()<p):
            lam = np.random.beta(1.5,1.5)
            item_mix = random.randint(0,len(self.image_list)-1)
            item_mix = self.image_list[item_mix]
            img_mix = cv2.imread(osp.join(self.image_path,item_mix))
            mask_mix  = cv2.imread(osp.join(self.mask_path,item_mix))
            """
                到这里完成了对应的挑选
            """
            augmentation_mix = self.augtransform(image = img_mix,mask = mask_mix)
            img_mix = augmentation_mix['image']
            mask_mix = augmentation_mix['mask']
            img = lam*img_org + (1-lam)*img_mix
            labelcol = mask_org + mask_mix
            return img,labelcol
        else:
            return img_org,mask_org
class SirVal(Dataset):
    def __init__(self,BaseDir):
        super().__init__()
        self.normalize = True
        self.base_dir = BaseDir
        self.image_path = osp.join(self.base_dir,'img')
        self.mask_path = osp.join(self.base_dir,'labelcol')
        self.image_list = os.listdir(self.image_path)
        self.transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
            ]
        )
        self.augtransform = A.Compose(
            [ 
                A.Resize(height = 320,width = 320,interpolation=cv2.INTER_CUBIC,p=1),
            ],
        )
    def __getitem__(self,idx):
        cur_img = self.image_list[idx]
        image = cv2.imread(osp.join(self.image_path,cur_img))
        mask  = cv2.imread(osp.join(self.mask_path,cur_img))
        augmentation = self.augtransform(image = image,mask = mask)
        image = augmentation['image']
        mask = augmentation['mask']
        if(self.normalize):
            image = image.astype(np.float32)/255.0
            mask = mask.astype(np.float32)/255.0   
        image = self.transform(image)
        mask = self.transform(mask)

        return image,mask

    def __len__(self):
        return len(self.image_list)