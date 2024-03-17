'''转换文件：将数据集中的dcm文件转换为png图片'''
import os
import SimpleITK as sitk
import cv2
import numpy as np

from CTAI_model.process import process


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# filename_list = make.get_person_files('../data_test/all/d2/')
filename_list = process.get_person_files('../data/tumor_data_train/')
for i in filename_list:
    pid = i[0]
    for j in i[1]:
        image = sitk.ReadImage(j)
        image_array = sitk.GetArrayFromImage(image).swapaxes(0, 2)
        image_array = np.rot90(image_array, -1)
        image_array = np.fliplr(image_array).squeeze()

        # ret, image_array = cv2.threshold(image_array, 150, 255, cv2.THRESH_BINARY)
        mkdir(f'../data/dcm_to_png_temp/{pid}/')
        name = j.replace('.dcm', '').split('/')[-1]
        cv2.imwrite(f'../data/dcm_to_png_temp/{pid}/{name}.png', image_array, (cv2.IMWRITE_PNG_COMPRESSION, 0))

print(filename_list)

# for i in filename_list:
#     if '.dcm' in i:
#         image = sitk.ReadImage(data_path + '/' + i)
#         image_array = sitk.GetArrayFromImage(image).swapaxes(0, 2)
#         image_array = np.rot90(image_rray, -1)
#         image_array = np.fliplr(image_array)
#         name = i.replace('.dcm', '')
#         cv2.imwrite(f'{data_path}/{name}_train.png', image_array, (cv2.IMWRITE_PNG_COMPRESSION, 0))

# t=cv2.imread('data_test/out/mask-tttt.png',cv2.IMREAD_GRAYSCALE)
# print(t)
