import Augmentor

'''
步骤 1.创建管道Pipeline实例
'''
# 设置原图所在目录：此处原图目录中不能有子文件夹，否则图片无法被读取到管道中！！！
p = Augmentor.Pipeline("../data/dcm_to_png_temp/tumor_data_train_x")
# p = Augmentor.Pipeline("../data/augment_data_test/original_picture")
print("管道状态:", p.status())
# 设置标签mask路径
p.ground_truth("../data/dcm_to_png_temp/tumor_data_train_y")
#
# '''
# 步骤 2：向管道添加数据增强（执行概率+参数范围）操作
# '''
# 设置处理概率
process_probability = 1
#
# 1、旋转
# 1.1 不带裁剪的旋转
# p.rotate_without_crop(probability=process_probability, max_left_rotation=60, max_right_rotation=60)
# 1.2 带裁剪的旋转
# p.rotate(probability=process_probability, max_left_rotation=25, max_right_rotation=25)
# 1.3 旋转90°
# p.rotate_random_90(probability=process_probability)
#
# 2、缩放
# p.zoom(probability=process_probability, min_factor=0.6, max_factor=1.5)
#
# # 3、裁剪
# # 3.1 按大小裁剪
# # p.crop_by_size(probability=1, width=1000, height=1000)
# # 3.2 从区域中心裁剪
# p.crop_centre(probability=1,percentage_area=0.8)
#
# # 3.3 随机裁剪
# p.crop_random(probability=1, percentage_area=0.8, randomise_percentage_area=True)
#
# 4、翻转
# 4.1 水平翻转
# p.flip_left_right(probability=process_probability)
#========================================================================
#
# 4.2 上下翻转
# p.flip_top_bottom(probability=process_probability)
#
# 4.3 随机翻转
# p.flip_random(probability=process_probability)
#
# 5、亮度增强/减弱
# p.random_brightness(probability=process_probability, min_factor=0.7, max_factor=1.2)
#========================================================================
#
# 6、颜色增强/减弱
p.random_color(probability=process_probability, min_factor=0.0, max_factor=1.5)
#
# 7、对比度增强/减弱
p.random_contrast(probability=process_probability, min_factor=0.7, max_factor=1.2)
#
# # 8、错切形变
# p.shear(probability=process_probability,max_shear_left=15,max_shear_right=15)
#
# # 9、透视形变
# 9.1 垂直方向透视形变
# p.skew_tilt(probability=process_probability, magnitude=1)
#
# # 9.2 斜四角方向透视形变
# p.skew_corner(probability=process_probability, magnitude=0.5)
#
# # 10、弹性扭曲
# p.random_distortion(probability=process_probability, grid_height=5, grid_width=16, magnitude=8)
#
# # 11、随机区域擦除
# p.random_erasing(probability=process_probability, rectangle_area=0.5)
#
#
# '''
# 步骤 3：生成数据增强后的图像和标签mask
# '''
# 设置生成个数
p.sample(20)
# 对每个图像做一次数据增强操作
p.process()
