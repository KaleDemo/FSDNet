from calendar import day_abbr
from statistics import mode
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.UNet_Dyhead import UNetDyhead,UNetDyheadv2,UNetBasic
from model.UNet3plus import UNet3Plus_DeepSup
from dataset.Sirst_Dataset import SirTrain,SirVal
from torchsummary import summary
from utils.metric import *
from utils.utils import save_ckpt,mkdir_exp
from torch.utils.data import DataLoader
from utils.logger import loadLogger
from loss.SoftiouLoss import SoftIoULoss
from loss.SoftiouLoss import AverageMeter
from utils.train_args import train_argparse
from model.hrnet import *
from loss import focal_Iou
from utils import cosine_lr_scheduler
from loss.focal_Iou import *
import os
seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
class Trainer():
    def __init__(self,args,log_path):
        self.args = args
        self.best_iou = 0
        self.best_niou = 0
        self.iou_metric = SigmoidMetric()
        self.niou_metric = SamplewiseSigmoidMetric(1,0.5)
        self.train_dataset = SirTrain(args.train_dataset)
        self.trainloader = DataLoader(dataset = self.train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
        self.val_dataset = SirVal(args.val_dataset)
        self.valloader = DataLoader(dataset=self.val_dataset,batch_size=1 , shuffle=True, num_workers=args.workers,drop_last=False)
        self.device = torch.device('cuda')
        self.model = UNetDyheadv2(deep_supervision=args.deep_supervision)
        self.model.cuda()
        self.criterion =  SoftIoULoss()
      # self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate,weight_decay=1e-5)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr = args.learning_rate, momentum=args.momentum,weight_decay=1e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = args.learning_rate,weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=20,T_mult=2,verbose=False)
        self.logger = loadLogger(self.args,path= log_path)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Total_params: {}".format(total_params))
        
    def training(self,epoch):
        tbar = tqdm(enumerate(self.trainloader))
        self.model.train()
        losses = AverageMeter()
        for i, (data, labels) in tbar:
            data = data.cuda()
            labels = labels.cuda()
            
            if self.args.deep_supervision:
               # print('deep supervision')
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    loss += self.criterion(pred,labels)
                loss /= len(preds)
            else:
                pred = self.model(data)
                loss = self.criterion(pred,labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(len(self.trainloader)*epoch + i)
            losses.update(loss.item(), pred.size(0))
        self.logger.info('epoch [{}/{}], loss:{:.5f}'
          .format(epoch,self.args.epochs, losses.avg))
        self.train_loss = losses.avg

    def testing(self,epoch,weight_path):
        tbar = tqdm(self.valloader)
        self.model.eval()
        losses = AverageMeter()
        self.iou_metric.reset() 
        self.niou_metric.reset()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                pred = self.model(data)
                middle_label = labels[:,0,:,:].unsqueeze(1)
                if self.args.deep_supervision:
               # print('deep supervision')
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += self.criterion(pred,labels)
                    loss /= len(preds)
                    pred = preds[0]
                else:
                    pred = self.model(data)
                    loss = self.criterion(pred,labels)
                losses.update(loss.item(), pred.size(0))
                self.iou_metric.update(pred,middle_label)
                self.niou_metric.update(pred,middle_label)
                _, IoU = self.iou_metric.get()
                _, nIoU = self.niou_metric.get()
                num += 1
                test_loss=losses.avg
        self.save_best_res(weight_path=weight_path,epoch=epoch,iou=IoU,niou=nIoU)
        self.logger.info('Epoch{} Best IoU: {}, best nIoU: {},eval loss {}'.format(epoch,self.best_iou,self.best_niou,losses.avg))
        
    def save_best_res(self,weight_path,epoch,iou=0.0,niou=0.0):
        """
        params: weight_path 权重路径名称
                iou niou
                还没有添加全局的log文件，和支持TensorBoardX明天
        return:
        """
        if iou > self.best_iou or niou > self.best_niou:
            save_path = os.path.join(os.path.join(weight_path,"weight_%s.pth"%(str(epoch))))
            if iou  > self.best_iou:
               self.best_iou = iou
               self.best_niou = niou
            if niou > self.best_niou:
               self.best_niou = niou
               self.best_iou = iou
            torch.save(self.model.state_dict(),save_path)

def main():
    """
    如何处理这个逻辑的关系的根据IoU或者nIoU的最大值保存对应的pth,断点重连应该放在一个较大的部分
    用来接续连接,类和类之间的耦合不能过于严重，否则会不太好改
    """
    args =  train_argparse().parse_args()
    start_epoch = 0
    ckpt_path = mkdir_exp('ckpt_EA')
    weight_path = mkdir_exp('weights_EA')
    log_path = mkdir_exp('log_EA')
    trainer = Trainer(args,log_path=log_path)
    if (args.RESUME):
        print('Resuming')
        path_checkpoint = args.ckpt_path
        checkpoint = torch.load(path_checkpoint)
        trainer.model.load_state_dict(checkpoint['net'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        trainer.scheduler.load_state_dict(checkpoint['lr_schedule'])
        print('finished')

    for epoch in range(start_epoch,args.epochs):
        trainer.training(epoch)
        if(epoch % trainer.args.ckpt_freq == 0):
            save_ckpt(epoch = epoch,ckpt_path=ckpt_path,trainer=trainer)
        trainer.testing(epoch,weight_path=weight_path)
if __name__ =='__main__':
    main()


     
 