from camera import one_frame_grab, distortion_correction
import cv2
import numpy as np
from chess_utils.logger import logger


def main():
    image = one_frame_grab()
    imgSource = image.copy()
    # imgSource = distortion_correction(imgSource)
    # 转为灰度图
    gray = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)

    # 中值滤波
    imgMedian = cv2.medianBlur(gray, 3)

    # 直方图均衡化
    imgEqualizeHist = cv2.equalizeHist(imgMedian)
    circles = cv2.HoughCircles(imgEqualizeHist, cv2.HOUGH_GRADIENT, dp=1, minDist=38, param1=91, param2=19,
                               minRadius=20, maxRadius=24)
    board_12_position_list = []  # 规定从左往右，从上往下
    if circles is not None:
        if len(circles[0]) != 12:
            logger.warning("标定识别的圆个数不等于12，请检查结果")
        for circles1 in circles[0]:
            x, y, r = int(circles1[0]), int(circles1[1]), int(circles1[2])
            # 画出圆的外接圆
            cv2.circle(image, (x, y), r, (0, 0, 255), 3)
            cv2.circle(imgSource, (x, y), r, (0, 0, 255), 3)
            # 画出圆心
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
            cv2.circle(imgSource, (x, y), 2, (0, 0, 255), 3)

            board_12_position_list.append([x, y])
        print(board_12_position_list)
        # 使用 sorted() 函数对二维数组进行排序，根据第二列的数据（即 index=1）
        sorted_2d_array = sorted(board_12_position_list, key=lambda x: x[1])

        # 提取前四个二维数组
        first_four_arrays = sorted_2d_array[:4]
        sorted_2d_array1 = sorted(first_four_arrays, key=lambda x: x[0])
        # 提取中间四个二维数组
        second_four_arrays = sorted_2d_array[4:8]
        sorted_2d_array2 = sorted(second_four_arrays, key=lambda x: x[0])
        # 提取最后四个二维数组
        last_four_arrays = sorted_2d_array[8:]
        sorted_2d_array3 = sorted(last_four_arrays, key=lambda x: x[0])
        # 将三个排序后的二维数组按顺序拼接起来
        final_sorted_array = sorted_2d_array1 + sorted_2d_array2 + sorted_2d_array3
        final_sorted_array = np.array(final_sorted_array, dtype=np.float32)
        print(final_sorted_array)
        # 要保存到的文件路径
        file_path = "checkerboard_12_pixels_position.npy"
        # 使用 numpy.save() 函数将数组保存为 .npy 文件
        np.save(file_path, final_sorted_array)
        print("checkerboard_12_pixels_position 已保存到文件:", file_path)
    cv2.imshow('imgSource', imgSource)
    cv2.imshow('ori', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
