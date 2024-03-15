import numpy as np
import cv2
import torch


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

    # dice = DiceLoss()
    # iou = SoftDiceLoss()
    #
    # # 自定义及计算dice损失
    # dice_value1 = dice_coef_loss(image1_tensor, image2_tensor)
    # print(dice_value1.item())
    # dice_value2 = iou(image1_tensor, image2_tensor)
    # print(dice_value2.item())
