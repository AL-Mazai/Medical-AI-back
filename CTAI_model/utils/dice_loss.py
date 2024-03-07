import numpy as np
import cv2


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # 俩都为全黑
    if not (im1.any() or im2.any()):
        return 1.0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    res = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return np.round(res, 5)


def read_image(file_path):
    image = cv2.imread(file_path)
    return image

if __name__ == '__main__':
    # 读取两张测试图片
    image1 = read_image(f'../../CTAI_flask/tmp/mask/20001_20240307155539_mask.png')
    image2 = read_image(f'../../CTAI_model/data/1002/venous phase/20001_mask.png')

    # 计算 Dice 系数
    dice_coefficient = dice(image1, image2)
    print("Dice coefficient:", dice_coefficient)
