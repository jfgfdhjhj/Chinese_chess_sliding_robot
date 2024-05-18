import cv2
import os

# 指定文件夹路径
folder_path = './images/'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为 jpg
    if filename.endswith('.jpg'):
        # 输出文件名
        print(filename)
        # 读取图片
        name_without_extension = os.path.splitext(filename)[0]
        print(name_without_extension)
        image = cv2.imread(folder_path + filename)
        # 获取图片中心点坐标

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 逐度旋转并输出图片
        for angle in range(0, 360, 1):
            # 旋转图片
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, M, (w, h))

            # 输出旋转后的图片，文件名包含旋转角度
            output_file = f'{filename[0]}.spin{angle:03d}(1).jpg'
            cv2.imwrite("spin" + "\\" + output_file, rotated_image)

            with open("lables/" + name_without_extension + ".txt", 'r') as f1:
                lines = f1.readlines()

            # 创建并写入txt文件
            txt_filename = f'lables/{filename[0]}.spin{angle:03d}(1).txt'
            with open(txt_filename, 'w') as f:
                f.writelines(lines)


