from cgi import test
from doctest import FAIL_FAST
import json
from sklearn.metrics import auc
from tqdm import tqdm
import torch.nn as nn
from model.UNet_Dyhead import UNetDyhead,UNetDyheadv2,UNetBasic
from model.UNet3plus import UNet3Plus_DeepSup
from dataset.Sirst_Dataset import SirTrain,SirVal
from torchsummary import summary
from utils.metric import *
from utils.utils import save_ckpt,mkdir_exp
from torch.utils.data import DataLoader
from utils.logger import loadLogger
import pandas as pd
import os.path as osp
import json
from utils.test_args import test_argparse
"""
    Test 部分主要实现的最佳功能如下：最佳权重加载，mask图片的生成和保存，必要函数的metric的计算方式，】
    Heatmap 下一步可视化的计算,综合以上的算法如下，进行简要的推理的计算。
    作为整体的权重进行对应的计算范式，主要参考以下的Demo库，
    完成如下的躬耕分别是Roc的调用
"""
class Tester(object):
    def __init__(self,args,roc_path,fa_pd_path,iou_niou_path):
        self.args = args
        self.ROC = ROCMetric(1, args.ROC_thr)
        self.PD_FA =  PD_FA(1,args.ROC_thr)
        self.test_dataset = SirVal(args.test_dataset)
        self.testloader =  DataLoader(dataset=self.test_dataset,batch_size=1 , shuffle=True, num_workers=args.workers,drop_last=False)
        self.roc_path = roc_path
        self.fa_pd_path = fa_pd_path
        self.iou_niou_path = iou_niou_path
        self.device = torch.device('cuda')
        self.iou_metric = SigmoidMetric()
        self.niou_metric = SamplewiseSigmoidMetric(1,0.5)
        self.SegMetric = SegmentationMetric(nclass = 1)

        self.model = UNetDyheadv2(deep_supervision=args.deep_supervision)
        self.model.cuda()
        self.model.load_state_dict(torch.load(args.best_weight_path))    
        print("The best weights has been loaded")
    @staticmethod
    def save_indicator(save_path,*result):   
        if(len(result)==4):
            roc_df = pd.DataFrame()
            roc_df['true_positive_rate'] = result[0]
            roc_df['flase_positive_rate'] = result[1]
            roc_df['recall'] = result[2]
            roc_df['precision'] = result[3]
            roc_df.to_csv(osp.join(save_path,'roc.csv'))
        elif(len(result)==2):
            fa_pd = pd.DataFrame()
            fa_pd ['FA'] = result[0]
            fa_pd ['PD'] = result[1]
            fa_pd.to_csv(osp.join(save_path,'fa_pd.csv'))
        else:
            """
                这里设计写入Json 文件当中
            """
            dict_demo = {'iou':result[0],'niou':result[1],'auc':result[2],
            'miou':result[3],'prec':result[4],'recall':result[5],'fmeasure':result[6]}
            print(dict_demo)
            jsobj  = json.dumps(dict_demo)
            file_object = open(osp.join(save_path,'IoU_json.json'),'w')
            file_object.write(jsobj)
            file_object.close()
    def testing(self):
        self.model.eval()
        tbar = tqdm(enumerate(self.testloader))
        with torch.no_grad():
            for  i,(data,labels) in tbar:
                data = data.cuda()
                labels = labels.cuda()
                if self.args.deep_supervision == True:
                    preds = self.model(data)
                    pred = preds[0]
                else:
                    pred = self.model(data)
                self.ROC.update(pred,labels)
                self.PD_FA.update(pred,labels)
                self.iou_metric.update(pred,labels)
                self.niou_metric.update(pred,labels)
                self.SegMetric.update(labels,pred)
            true_positive_rate,false_postive_rate,recall,precision = self.ROC.get()
            auc_value = auc(false_postive_rate,true_positive_rate)
            FA,PD =  self.PD_FA.get(len(self.test_dataset))
            _, iou = self.iou_metric.get()
            _, n_iou = self.niou_metric.get()
            miou, prec, recall_item, fmeasure = self.SegMetric.get()
            ## 这边再保存一个iou,niou的数值
            ## 大概就是这样每一个指标进行对应的分开保存
            if(self.args.csv_flag):
                Tester.save_indicator(self.roc_path,true_positive_rate,false_postive_rate,recall,precision)
                print('roc indicator has been saved')
                Tester.save_indicator(self.fa_pd_path,FA,PD)
                print('false alarm and PD have been saved')
                Tester.save_indicator(self.iou_niou_path,iou,n_iou,auc_value,miou,prec, recall_item,fmeasure)  # 本质上是一种占位符
                print('iou and niou have been saved')

     
def main():
    """
    如何处理这个逻辑的关系的根据IoU或者nIoU的最大值保存对应的pth,断点重连应该放在一个较大的部分
    用来接续连接,类和类之间的耦合不能过于严重，否则会不太好改
    """
    args =  test_argparse().parse_args()
    fa_pd_path = mkdir_exp('indicator','PD_FA')
    roc_path = mkdir_exp('indicator','ROC')
    iou_niou = mkdir_exp('indicator','iou_niou')
    tester = Tester(args,roc_path=roc_path,fa_pd_path=fa_pd_path,iou_niou_path = iou_niou)
    tester.testing()

    
if __name__ =='__main__':
    main()


    









