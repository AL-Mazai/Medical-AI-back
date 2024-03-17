import os
import shutil

# 源文件夹和目标文件夹路径
source_folder = '../data/tumor_data_train/output_new'
target_folder = '../data/tumor_data_train/data'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# for filename in os.listdir(source_folder):
#     if '_mask' not in filename:
#         # 构造源文件和目标文件的完整路径
#         source_path = os.path.join(source_folder, filename)
#         target_path = os.path.join(target_folder, filename)
#         # 移动文件到目标文件夹
#         shutil.move(source_path, target_path)


# for filename in os.listdir(source_folder):
#     source_path = os.path.join(source_folder, filename)
#     target_path = os.path.join(target_folder, filename)
#     # 移动文件到目标文件夹
#     shutil.move(source_path, target_path)


# 复制
for filename in os.listdir(source_folder):
    source_path = os.path.join(source_folder, filename)
    target_path = os.path.join(target_folder, filename)
    # 移动文件到目标文件夹
    shutil.copy(source_path, target_path)
