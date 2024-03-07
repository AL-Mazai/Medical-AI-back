import sys

# sys.path.append("..")
import cv2
import torch
from torch.nn import init
from torch.utils.data import DataLoader
from CTAI_model.data_process import make
from CTAI_model.net import unet
from CTAI_model.utils import dice_loss
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
res = {'epoch': [], 'loss': [], 'dice': [], 'val_dice': []}


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)


# 参数
rate = 0.50  # 计算dice系数的阈值
learn_rate = 0.001
epochs = 100
# train_dataset_path = '../data/all/d1/'
train_dataset_path = '../data/1002/'
train_dataset, test_dataset = make.get_d1(train_dataset_path)

unet = unet.Unet(1, 1).to(device).apply(weights_init)
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(unet.parameters(), learn_rate)


def train():
    global res
    dataloaders = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    for epoch in range(epochs):
        dt_size = len(dataloaders.dataset)
        epoch_loss, epoch_dice = 0, 0
        step = 0
        for x, y in dataloaders:
            id = x[1:]
            step += 1
            x = x[0].to(device)
            y = y[1].to(device)
            # TODO
            y = y.reshape(2, 1, 512, 512)
            # print(x.size())
            # print(y.size())
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
            epoch_dice += dice

            # if step % 10 == 0:
            # res['epoch'].append((epoch + 1) * step)
            # res['loss'].append(loss.item())
            # print("epoch %d step%d/%d train_loss:%0.3f" % (
            #     epoch + 1, step, (dt_size - 1) // dataloaders.batch_size + 1, loss.item()), end='\n')
            # test()

        res['epoch'].append(epoch + 1)
        res['loss'].append(loss.item())
        res['dice'].append(epoch_dice / step)
        print("epoch %d loss:%0.3f,dice:%f" % (epoch + 1, epoch_loss / step, epoch_dice / step))
        validate()

    # 保存模型
    torch.save(unet.state_dict(), '../../CTAI_flask/core/net/model.pth')
    # 可视化
    plt.plot(res['epoch'], np.squeeze(res['loss']), label='Train loss')
    plt.plot(res['epoch'], np.squeeze(res['dice']), label='Train dice', color='orange')
    plt.plot(res['epoch'], np.squeeze(res['val_dice']), label='Validate dice', color='red')

    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def validate():
    global res, img_y, mask_arrary
    epoch_dice = 0
    with torch.no_grad():
        dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
        for x, mask in dataloaders:
            id = x[1:]  # ('1026',), ('10018',)]先病人号后片号
            x = x[0].to(device)
            y = unet(x)
            mask_arrary = mask[1].cpu().squeeze(0).detach().numpy()
            img_y = torch.squeeze(y).cpu().numpy()
            img_y[img_y >= rate] = 1
            img_y[img_y < rate] = 0
            img_y = img_y * 255

            epoch_dice += dice_loss.dice(img_y, mask_arrary)
            cv2.imwrite(f'../data/out/{mask[0][0]}-result.png', img_y, (cv2.IMWRITE_PNG_COMPRESSION, 0))
        print('val dice:%f' % (epoch_dice / len(dataloaders)))
        res['val_dice'].append(epoch_dice / len(dataloaders))


if __name__ == '__main__':
    train()
