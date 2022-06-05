# FSDNet
# DataSet
 - SIRST
# BackBone
 - UNet++
 - HRNet
 - UNet+++
 - UNet
# Neck
 - Bi-FPN
 - FPN
 - Dynamic Head
# Segmentation Head
 - CBAM Head
# Loss
 -  BCE Diceloss
 -  LogNLLoss
 -  FocalLoss
 -  SoftIoU loss
 -  FocalSoftLoss 
 -  Lovasoftmax
# Data Augmentation
 - Crop \& Paste
 - Mix up
 - Colorjiter
 - oversampling
# Method FSDNet
# How to start
please refer to detectron.yaml to install necessary package. For the Dynamic head, you can refer the Dynamic Head to implementation. All the loss have been conducted on the UNetDynamic. Overall structure refer the DNANet. If you need use this module, please cite the following bibtex. 
 
```
@article{DBLP:journals/corr/abs-2106-00487,
  title = {Dense Nested Attention Network for Infrared Small Target Detection},
  author = {Boyang Li and
               Chao Xiao and
               Longguang Wang and
               Yingqian Wang and
               Zaiping Lin and
               Miao Li and
               Wei An and
               Yulan Guo},
  year = {2021},
  journal = {arXiv preprint arXiv:2106.00487},
  eprint = {2106.00487},
  eprinttype = {arxiv},
  archiveprefix = {arXiv}
}
```
.
