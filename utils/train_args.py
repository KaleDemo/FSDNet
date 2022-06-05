import argparse
import os
"""
To do:
    大量的冗余命令行参数还没有完全除去，我先能够跑起来再说,其中数据增强以及相应的collate_fn参数
    以及对应的数据增强的方面，需要借鉴一下其他人的代码并且能正常的进行，全局预处理的方向上可以按照
    ACM和DNANet 两个不错的开源增强库，两个模块以及相应的基于Filter或者LCM
"""
def train_argparse():
    parser = argparse.ArgumentParser(description='MPANet train process')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default= 500, type=int, metavar='N',
                        help='number of total epochs to run(default: 400)')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default = 8, type=int,
                        metavar='N', help='batch size (default: 1)')
    parser.add_argument('--learning_rate', default=1.5e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--train_dataset', type=str,default='/home/wa/ENHANCE/generate')
    parser.add_argument('--val_dataset', type=str,default='/home/wa/ENHANCE/test')
    parser.add_argument('--save_freq', type=int,default =120)
    parser.add_argument('--ckpt_freq', type=int, default=80)
    parser.add_argument('--ckpt_path', default='/home/wa/MPANet/ckpt_aug/exp2/ckpt_pat240.pth', type=str,
                        help='directory to save')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--RESUME',default=False,type=bool,help = 'checkpoint')
    parser.add_argument('--deep_supervision',default= True,type = bool,help ='deep supervision')

    return parser