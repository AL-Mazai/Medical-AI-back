import numpy as np
import cv2
import torch
import torch.nn.functional as F


def dice(predict, target):
    predict = np.asarray(predict).astype(bool)
    target = np.asarray(target).astype(bool)

    if predict.shape != target.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # 俩都为全黑
    t1 = predict.any()
    t2 = target.any()
    if not (predict.any() or target.any()):
        return 1.0

    # Compute Dice coefficient
    # intersection = np.logical_and(predict, target)
    intersection = (predict * target).sum()
    res = 2. * intersection.sum() / (predict.sum() + target.sum())
    return np.round(res, 5)


ALPHA = 0.6
GAMMA = 2
class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss

def read_image(file_path):
    image = cv2.imread(file_path)
    return image


if __name__ == '__main__':
    # 读取两张测试图片
    image1 = read_image(f'../../CTAI_model/data/1002/venous phase/20015_mask.png')
    image2 = read_image(f'../../CTAI_model/data/1002/venous phase/20014_mask.png')

    # 计算 Dice 系数
    dice_coefficient = dice(image1, image2)

    print("Dice coefficient:", dice_coefficient)

    # # # 将 NumPy 数组转换为 PyTorch 张量
    image1_tensor = torch.from_numpy(image1).float()
    image2_tensor = torch.from_numpy(image2).float()

    dice = FocalLoss()
    # iou = SoftDiceLoss()
    #
    # # 自定义及计算dice损失
    dice_value1 = dice(image1_tensor, image2_tensor)
    print(dice_value1.item())
    # dice_value2 = iou(image1_tensor, image2_tensor)
    # print(dice_value2.item())
