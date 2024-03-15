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

test_data_path = '../data/test/'
test_dataset = make.get_d1_local(test_data_path)
# 计算dice系数的阈值
rate = 0.5


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        if not (inputs.any() or targets.any()):
            return 1.0
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice.item()


def test():
    unet = Unet(1, 1).to(device)
    unet.load_state_dict(torch.load('model.pth'))

    global res, predict, target
    epoch_dice = 0
    with torch.no_grad():
        dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        for x, mask in dataloaders:
            id = x[1:]  # ('1026',), ('10018',)]先病人号后片号
            # print(id, 'id')
            x = x[0].to(device)
            y = unet(x)

            # dice
            target = mask[1].cpu().squeeze(0).detach().numpy()
            predict = torch.squeeze(y).cpu().numpy()
            predict[predict >= rate] = 1
            predict[predict < rate] = 0
            predict = predict * 255

            dice = dice_loss.dice(predict, target)
            # if dice == 0:
            #     epoch_dice += 1.0
            #     print("dice:%f" % 1.0)
            # else:
            epoch_dice += dice
            print("dice:%f" % dice_loss.dice(predict, target))
            # cv2.imwrite(f'../data/out/{id[0][0]}/arterial phase/{id[1][0]}_mask.png', predict,
            #             (cv2.IMWRITE_PNG_COMPRESSION, 0))

    print("avg_dice:%f" % (epoch_dice / len(dataloaders)))
    # mkdir(f'../data/out/{id[0][0]}/arterial phase/')


if __name__ == '__main__':
    test()
