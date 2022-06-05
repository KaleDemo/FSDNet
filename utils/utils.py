import os
import cv2
import pandas as pd
import torch
import datetime
def mkdir_exp(*property):
    """
    :param property: 输入的是当前的文件名
    :return:文件的返回地址，最多支持二级地址的索引,然后就可以了
    """
    if(len(property)==1):
        property_path = os.path.join(os.getcwd(), property[0])
    else:
        property_path = os.path.join(os.getcwd(),property[0],property[1])
    if (not os.listdir(property_path)):
        cur_path = os.path.join(property_path, 'exp1')
        os.mkdir(cur_path) 
    else:
        cur_sequence = sorted(list(map(lambda x:int(x[3:]),os.listdir(property_path))))
        cur_path = os.path.join(property_path, 'exp' + str(cur_sequence[-1]+ 1))
        os.mkdir(cur_path)
    return cur_path
def save_ckpt(epoch,ckpt_path,trainer,filename = None):
    if filename:
        save_path = os.path.join(os.path.join(ckpt_path,"%s_ckpt_%s.pth"%(filename,str(epoch))))
    else:
        save_path = os.path.join(os.path.join(ckpt_path,"ckpt_pat%s.pth"%(str(epoch))))
    checkpoint = {"net": trainer.model.state_dict(),
                  "optimizer": trainer.optimizer.state_dict(),
                  'epoch': epoch,
                  'lr_schedule':trainer.scheduler.state_dict()}
    torch.save(checkpoint,save_path)

def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision ):
    with open(dataset_dir + '/' + 'value_result'+'/' + st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open(dataset_dir + '/' +'value_result'+'/'+ st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return