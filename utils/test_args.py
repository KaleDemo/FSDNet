import argparse
import os
from xml.etree.ElementInclude import default_loader
"""
To do:
    大量的冗余命令行参数还没有完全除去，我先ome/wa/MPANet/ckpt_aug/exp2/ckpt_pat240.pth能够跑起来再说,其中数据增强以及相应的collate_fn参数
    以及对应的数据增强的方面，需要借鉴一下其他人的代码并且能正常的进行，全局预处理的方向上可以按照
    ACM和DNANet 两个不错的开源增强库，两个模块以及相应的基于Filter或者LCM
"""
def test_argparse():
    parser = argparse.ArgumentParser(description='UNetDynamicV2 QAQ')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--test_dataset', type=str,default='/home/wa/ENHANCE/test')
    parser.add_argument('--best_weight_path', default='/home/wa/UNetDynamic/weights_EA/exp45/weight_123.pth', type=str,
                        help='load best pth')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--csv_flag',type = bool,default = True,help = 'roc_csv_flag 标志')
    parser.add_argument('--deep_supervision',default=True,type = bool,help = 'model seletion')
    parser.add_argument('--ROC_thr', default = 50, type = int, help = 'roc bins number')
    return parser