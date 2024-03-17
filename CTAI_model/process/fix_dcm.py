import os
import pydicom

# 源文件夹和目标文件夹路径
source_folder = '../data/tumor_data_train/augment_data'
target_folder = '../data/tumor_data_train/augment_data'

patient_pid = 20230726001
accession_number = 202307261001
study_uid = 2023072620001
seriesNumber = 1
seriesInstanceUID = "1.2.410.200048.2858.20230529094313.1"
modality = "CR"
pixelSpacing = [0.160145, 0.160114]
instanceNumber = 1
bodyPartExamined = "CHEST"

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    if filename.endswith('.dcm'):
        # 构建源文件路径和目标文件路径
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)

        # 加载源DCM文件
        dcm_data = pydicom.dcmread(source_file, force=True)

        # 添加患者PID、Accession Number和Study UID等信息
        dcm_data.PatientID = str(patient_pid)
        dcm_data.AccessionNumber = str(accession_number)
        dcm_data.StudyInstanceUID = str(study_uid)
        dcm_data.SeriesNumber = seriesNumber
        dcm_data.SeriesInstanceUID = seriesInstanceUID
        dcm_data.Modality = modality
        dcm_data.PixelSpacing = pixelSpacing
        dcm_data.BodyPartExamined = bodyPartExamined
        dcm_data.InstanceNumber = instanceNumber

        # 将文件名作为患者名
        file_name_without_extension = os.path.splitext(filename)[0]
        dcm_data.PatientName = file_name_without_extension

        # 保存修改后的DCM文件到目标文件夹
        dcm_data.save_as(target_file)

        # 递增计数器
        patient_pid += 1
        accession_number += 1
        study_uid += 1
    else:
        print("error!")
