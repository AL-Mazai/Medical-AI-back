import os
import shutil

# 源文件夹和目标文件夹路径
source_folder = '../data/pnd_to_dcm_temp/output'
target_folder = '../data/tumor_data_train/augment_data'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的文件
# for filename in os.listdir(source_folder):
#     # 检查文件名是否不包含 '_mask'
#     if '_mask' not in filename:
#         # 构造源文件和目标文件的完整路径
#         source_path = os.path.join(source_folder, filename)
#         target_path = os.path.join(target_folder, filename)
#         # 移动文件到目标文件夹
#         shutil.move(source_path, target_path)


# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 构造源文件和目标文件的完整路径
    source_path = os.path.join(source_folder, filename)
    target_path = os.path.join(target_folder, filename)
    # 移动文件到目标文件夹
    shutil.move(source_path, target_path)
