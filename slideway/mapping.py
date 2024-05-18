import json
import math
import cv2
import numpy as np
from system import board_matrix, slideway_coordinates_dict, chess_board_dic, slideway_coordinates_path
import os
from chess_utils.logger import logger

# 单个横着的棋盘宽 mm
dar_x = 25.5
# 单个横着的棋盘高 mm
dar_y = 24.1
# 原点距离棋盘竖着的右上角绝对值 mm
offset_y = 37.5


def generate_slide_coordinates(num_rows=9, num_columns=10):
    ori = (0, -offset_y)
    top_left = (ori[0] + dar_y * 8, ori[1] - dar_x * 9)
    # 计算其他点的坐标
    points1 = []
    i = 0
    for row in range(num_rows):
        j = 0
        for column in range(num_columns):
            x = int(top_left[0] - row * dar_y)
            y = int(top_left[1] + column * dar_x)
            points1.append((i, j, x, y))
            j += 1
        i += 1

    with open(slideway_coordinates_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(points1))
        print("滑轨坐标保存完成!")

    return points1


def slideway_four_coordinate_return():
    x1, y1 = slideway_coordinates_dict[0][2], slideway_coordinates_dict[0][3]
    x2, y2 = slideway_coordinates_dict[9][2], slideway_coordinates_dict[9][3]
    x3, y3 = slideway_coordinates_dict[89][2], slideway_coordinates_dict[89][3]
    x4, y4 = slideway_coordinates_dict[80][2], slideway_coordinates_dict[80][3]

    slide_four_array = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)
    return slide_four_array


class HandInEyeCalibrationSlide:
    origin = (0, 0)
    origin_to_the_left = (200, 0)

    def __init__(self):
        self.matrix = None
        self.STC_points_camera = np.array([chess_board_dic["chess_4position"][i] for i in range(4)], dtype=np.float32)
        # logger.debug(self.STC_points_camera)
        self.STC_points_slideway = slideway_four_coordinate_return()
        points_camera = self.trans_board_affine_point(self.STC_points_camera)
        self.matrix = cv2.getPerspectiveTransform(points_camera, self.STC_points_slideway)

    def trans_board_affine_point(self, points_camera):
        # 使用透视变换矩阵对坐标进行变换,此透视变换针对图片的点进行校正
        points_camera = points_camera.astype(np.float32)  # 将数据类型转换为32位浮点型
        points_camera = cv2.perspectiveTransform(points_camera.reshape(-1, 1, 2), board_matrix)
        # logger.debug("透视变换的结果为：{}".format(points_camera))
        return points_camera

    def _get_points_slide(self, x_camera, y_camera):
        """
        相机坐标通过仿射矩阵变换取得滑轨坐标
        :param x_camera:
        :param y_camera:
        :return:
        """
        # 定义要映射的单个点,此点不经过任何处理，仅仅为图片识别的像素点
        point = np.array([x_camera, y_camera], dtype=np.float32)
        # 进行透视变换将点变成垂直的点
        point = self.trans_board_affine_point(point)
        # 应用透视变换将图片的坐标变换到滑轨的坐标上
        mapped_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.matrix)

        slide_x = int(mapped_point[0][0][0])
        slide_y = int(mapped_point[0][0][1])
        logger.info("映射出的滑轨坐标为：robot_x:{}, robot_y:{}".format(slide_x, slide_y))
        return slide_x, slide_y

    def get_points_slide_limit_x_y(self, slide_x, slide_y):
        """
        滑轨限位
        """
        slide_x, slide_y = self._get_points_slide(slide_x, slide_y)
        if -10 <= slide_x <= 202 or -277 <= slide_y <= 15:
            logger.info("滑轨映射出的坐标符合范围")
            return True, slide_x, slide_y
        else:
            logger.warning("滑轨映射出的坐标不符合范围")
            return False


if __name__ == "__main__":
    slide_coordinate = HandInEyeCalibrationSlide()
    res = slide_coordinate.get_points_slide_limit_x_y(1370, 935)
    if res[0]:
        print(res[1], res[2])
