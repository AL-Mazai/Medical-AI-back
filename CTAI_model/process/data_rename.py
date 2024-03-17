import os
import re

# 输入目录
input_dir = "../data/dcm_to_png_temp/tumor_data_train_x/output"
# 输出目录
output_dir = "../data/augment_data_test/output_new"

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录
for filename in os.listdir(input_dir):
    # 匹配肿瘤图片文件名格式
    # tumor_pattern = r"original_picture_original_(\d+)_([a-f0-9]+)\.png"
    # tumor_pattern = r"original_picture_original_(\d+).png_([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)\.png"
    tumor_pattern = r"tumor_data_train_x_original_(\d+)_([a-z ]+)_(\d+).png_([a-z0-9-]+)\.png"

    tumor_match = re.match(tumor_pattern, filename)
    if tumor_match:
        # x = "tumor_data_train_x_original_1064_venous phase_20029.png_9db1de9a-f158-445e-94b6-ce98f19c559c.png"
        # y = "_groundtruth_(1)_tumor_data_train_x_1080_venous phase_20013.png_c8e999ea-ce15-4294-a747-5ae6af0874e2.png"
        # 提取文件名中的后缀
        suffix = tumor_match.group(4)
        # suffix1 = tumor_match.group(3)
        # suffix2 = tumor_match.group(4)
        # suffix3 = tumor_match.group(5)
        # suffix4 = tumor_match.group(6)
        # suffix = suffix0 + '-' + suffix1 + '-' + suffix2 + '-' + suffix3 + '-' + suffix4

        # 构建肿瘤图片文件路径和对应的标签文件路径
        tumor_file_path = os.path.join(input_dir, filename)
        label_filename = f"_groundtruth_(1)_tumor_data_train_x_{tumor_match.group(1)}_{tumor_match.group(2)}_{tumor_match.group(3)}.png_{suffix}.png"
        label_file_path = os.path.join(input_dir, label_filename)
        source_filename = f"tumor_data_train_x_original_{tumor_match.group(1)}_{tumor_match.group(2)}_{tumor_match.group(3)}.png_{suffix}.png"
        source_file_path = os.path.join(input_dir, source_filename)

        # 构建重命名后的标签文件名
        new_source_filename = f"{suffix}.png"
        new_source_file_path = os.path.join(output_dir, new_source_filename)
        new_label_filename = f"{suffix}_mask.png"
        new_label_file_path = os.path.join(output_dir, new_label_filename)

        # 重命名数据文件和标签文件
        os.rename(label_file_path, new_label_file_path)
        os.rename(source_file_path, new_source_file_path)

        print(f"Renamed label file: {label_file_path} -> {new_label_file_path}")
    else:
        print(f"No match found for file: {filename}")

print("Finished renaming files.")
