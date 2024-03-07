import os
import sys
import cv2
# sys.path.append("..")
import torch
from torch.utils.data import DataLoader
from CTAI_model.data_process import make
from CTAI_model.net.unet import Unet
from CTAI_model.utils import dice_loss
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
res = {'epoch': [], 'loss': [], 'dice': []}

test_data_path = '../data/1002/'
test_dataset = make.get_d1_local(test_data_path)
# 计算dice系数的阈值
rate = 0.5


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def test():
    unet = Unet(1, 1).to(device)
    unet.load_state_dict(torch.load('model.pth'))

    global res, img_y, mask_arrary
    epoch_dice = 0
    epoch = 0
    with torch.no_grad():
        dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        for x, mask in dataloaders:
            epoch = epoch + 1
            id = x[1:]  # ('1026',), ('10018',)]先病人号后片号
            # print(id, 'id')
            x = x[0].to(device)
            y = unet(x)

            # dice
            mask_arrary = mask[1].cpu().squeeze(0).detach().numpy()
            img_y = torch.squeeze(y).cpu().numpy()
            img_y[img_y >= rate] = 1
            img_y[img_y < rate] = 0
            img_y = img_y * 255

            epoch_dice += dice_loss.dice(img_y, mask_arrary)
            print("dice:%f" % (epoch_dice / len(dataloaders)))
            # mkdir(f'../data/out/{id[0][0]}/arterial phase/')
            # cv2.imwrite(f'../data/out/{id[0][0]}/arterial phase/{id[1][0]}_mask.png', img_y,
            #             (cv2.IMWRITE_PNG_COMPRESSION, 0))


if __name__ == '__main__':
    test()
