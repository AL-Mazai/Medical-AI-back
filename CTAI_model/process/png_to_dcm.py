import os
import pydicom
from PIL import Image


def png_to_dicom(input_png_path, output_dcm_path, patient_name="Anonymous", study_description="PNG to DICOM"):
    for fileNames in os.listdir(input_png_path):
        input_filename = os.path.basename(fileNames).split('.')[0]
        output_filename = input_filename + ".dcm"
        input_filepath = input_png_path + fileNames
        output_dcmpath = output_dcm_path + output_filename

        # 读取PNG图像
        img = Image.open(input_filepath)

        # 将PNG图像转换为灰度图像（单通道）
        pixel_array = img.convert("L")

        # 创建一个空的FileDataset对象，并添加DICOM数据集元素
        ds = pydicom.dataset.FileDataset(output_dcm_path, {}, file_meta=pydicom.dataset.Dataset())  # 创建文件元信息头对象
        # 添加DICOM文件元信息头
        ds.file_meta.FileMetaInformationGroupLength = 184
        ds.file_meta.FileMetaInformationVersion = b'\x00\x01'
        ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'
        ds.file_meta.MediaStorageSOPInstanceUID = '1.2.410.200048.2858.20230531153328.1.1.1'
        ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
        ds.file_meta.ImplementationClassUID = '1.2.276.0.7230010.3.0.3.5.4'
        ds.file_meta.ImplementationVersionName = 'ANNET_DCMBK_100'

        # 添加DICOM数据集元素
        ds.PatientName = patient_name
        ds.StudyDescription = study_description
        ds.Columns, ds.Rows = img.size
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        # 数据显示格式
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = pixel_array.tobytes()  # 直接使用灰度图像的字节数据

        # 保存DICOM数据集到文件
        ds.is_little_endian = True
        ds.is_implicit_VR = True  # 使用隐式VR

        ds.save_as(output_dcmpath)
        print(output_dcmpath)


if __name__ == "__main__":
    # 输入PNG图像路径和输出DICOM图像路径
    input_png_path = "../data/pnd_to_dcm_temp/input/"
    output_dcm_path = "../data/pnd_to_dcm_temp/output/"

    # 将PNG转换为DICOM
    png_to_dicom(input_png_path, output_dcm_path)
