import pydicom

# 读取DICOM文件
dicom_file_path = '../data/tumor_data_train/augment_data/0a2e1b8e-2d0e-4dbc-867d-f2152efc916c.dcm'
# dicom_file_path = '../data/tumor_data_train/data_set_modify/1001_arterial phase_10007.dcm'
# dicom_file_path = '../data/tumor_data_train/data_set_modify/1049_arterial phase_10014.dcm'
# dicom_file_path = '../data/tumor_data_train/data_set_modify/1001_venous phase_20011.dcm'
ds = pydicom.dcmread(dicom_file_path, force=True)

# 输出DICOM文件的元数据信息
print("DICOM文件元数据信息：")
print(ds)

# 输出DICOM文件的标签（Tag）信息
# print("DICOM文件标签（Tag）信息：")
# for elem in ds:
#     print(elem.tag, "-", elem.description)
