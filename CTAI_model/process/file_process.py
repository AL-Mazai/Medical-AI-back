import os
import shutil


# 定义函数，删除文件夹及其内容
def delete_folder(folder_path):
    try:
        # 删除文件夹及其内容
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def move_files(source_folder, destination_folder):
    """
    将源文件夹中的所有文件移动到目标文件夹中。

    参数：
        source_folder：源文件夹的路径。
        destination_folder：目标文件夹的路径。
    """
    # 获取源文件夹中的文件列表
    files_to_move = os.listdir(source_folder)
    # 移动每个文件到目标文件夹
    for file_name in files_to_move:
        # 构建源文件路径和目标文件路径
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)
        # 移动文件
        shutil.move(source_file_path, destination_file_path)
    print("Files moved successfully.")


# 定义函数，遍历文件夹并重命名文件
def rename_files(root_dir):
    # 遍历直肠癌数据文件夹下的所有子文件夹
    for subdir_one in os.listdir(root_dir):
        if subdir_one != "1001":
            subdir = os.path.join(root_dir, subdir_one)
            move_files(subdir, "D:/大学/课程/大三下/软件创新/期末大作业/直肠癌数据_修改/1001")

        for subdir_two, dirs, files in os.walk(subdir):
            # 获取子文件夹的名称（例如1001）
            folder_name = os.path.basename(subdir_two)
            # 如果子文件夹中有arterial phase和venous phase两个文件夹
            if "arterial phase" in dirs or "venous phase" in dirs:
                # 遍历这两个文件夹
                for phase_dir in ["arterial phase", "venous phase"]:
                    phase_path = os.path.join(subdir, phase_dir)
                    delete_folder(phase_path)  # 删除文件夹
                    # 遍历当前文件夹下的所有文件
                    for file in os.listdir(phase_path):
                        # 构建新的文件名，添加相应的路径名前缀
                        new_name = os.path.join(subdir, f"{folder_name}_{phase_dir}_{file}")
                        # 重命名文件
                        os.rename(os.path.join(phase_path, file), new_name)
                        print(f"Renamed {file} to {new_name}")


# 调用函数并传入直肠癌数据文件夹的路径
root_directory = "D:/大学/课程/大三下/软件创新/期末大作业/直肠癌数据_修改"
rename_files(root_directory)
