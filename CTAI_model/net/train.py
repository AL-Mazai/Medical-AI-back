import sys

# sys.path.append("..")
import cv2
import torch
from torch.nn import init
from torch.utils.data import DataLoader
from CTAI_model.process import process
from CTAI_model.net import unet
from CTAI_model.utils import dice_loss
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
res = {'epoch': [], 'loss': [], 'dice': [], 'val_dice': [], 'val_loss': []}


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)


class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


# 参数
rate = 0.50  # 计算dice系数的阈值
learn_rate = 0.001
batch_size = 2
epochs = 2
# train_dataset_path = '../data_test/all/d1/'
train_dataset_path = '../data/1001/'
train_dataset, test_dataset = make.get_d1(train_dataset_path)

unet = unet.Unet(1, 1).to(device).apply(weights_init)
criterion = torch.nn.BCELoss().to(device)
# criterion = DiceBCELoss().to(device)
optimizer = torch.optim.Adam(unet.parameters(), learn_rate)


def train():
    global res
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for epoch in range(epochs):
        epoch_loss, epoch_dice = 0, 0
        best_dice = 0
        step = 0
        for x, y in dataloaders:
            id = x[1:]
            step += 1
            x = x[0].to(device)
            y = y[1].to(device)
            # TODO
            # y = y.reshape(2, 1, 512, 512)
            y = y.unsqueeze(1)
            optimizer.zero_grad()
            outputs = unet(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # loss
            epoch_loss += float(loss.item())
            # dice
            a = outputs.cpu().detach().squeeze(1).numpy()
            a[a >= rate] = 1
            a[a < rate] = 0
            b = y.cpu().detach().squeeze(1).numpy()
            dice = dice_loss.dice(a, b)
            # print("dice:%f", dice)
            epoch_dice += dice

            if step % 50 == 0:
                print("dice:%f", epoch_dice / step)


        res['epoch'].append(epoch + 1)
        res['loss'].append(loss.item())
        res['dice'].append(epoch_dice / step)

        # 保存dice最好的模型
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(unet.state_dict(), '../../CTAI_model/net/model.pth')

        print("epoch %d loss:%0.3f,dice:%f" % (epoch + 1, epoch_loss / step, epoch_dice / step))
        # 验证
        validate()


    # 可视化
    plt.plot(res['epoch'], np.squeeze(res['loss']), label='Train loss')
    plt.plot(res['epoch'], np.squeeze(res['dice']), label='Train dice', color='orange')

    plt.plot(res['epoch'], np.squeeze(res['val_dice']), label='Validate dice', color='red')
    plt.plot(res['epoch'], np.squeeze(res['val_loss']), label='Validate loss', color='yellow')

    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend()

    # 保存图形
    plt.savefig('result.png')
    plt.show()


def validate():
    global res, img_y, mask_arrary
    epoch_dice = 0
    epoch_loss = 0

    with torch.no_grad():
        dataloaders = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for x, mask in dataloaders:
            # id = x[1:]  # ('1026',), ('10018',)]先病人号后片号
            x = x[0].to(device)
            y = unet(x)

            # loss
            # mask[1] = mask[1].reshape(2, 1, 512, 512).to(device)
            mask[1] = mask[1].unsqueeze(1).to(device)
            loss = criterion(y, mask[1])
            epoch_loss += float(loss.item())

            # dice
            a = y.cpu().detach().squeeze(1).numpy()
            a[a >= rate] = 1
            a[a < rate] = 0
            a = a * 255
            b = mask[1].cpu().squeeze(1).detach().numpy()
            epoch_dice += dice_loss.dice(a, b)
            # cv2.imwrite(f'../data_test/out/{mask[0][0]}-result.png', img_y, (cv2.IMWRITE_PNG_COMPRESSION, 0))

        print('val loss:%f ,val dice:%f' % (epoch_loss / len(dataloaders), epoch_dice / len(dataloaders)))
        res['val_dice'].append(epoch_dice / len(dataloaders))
        res['val_loss'].append(epoch_loss / len(dataloaders))


if __name__ == '__main__':
    train()
