import os
import shutil

# 源文件夹和目标文件夹路径
source_folder = '../data/tumor_data_train/data_set_modify'
target_folder = '../data/dcm_to_png_temp/tumor_data_train_y'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 检查文件名是否包含 '_mask'
    if '_mask' in filename:
        # 去除 '_mask'
        new_filename = filename.replace('_mask', '')
        # 构造源文件和目标文件的完整路径
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, new_filename)
        # 复制文件到目标文件夹
        shutil.copyfile(source_path, target_path)
