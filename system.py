import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from chess_utils.logger import logger

# 定义各个路径部分
base_dir = "chess_dev"
src_dir = "src"
engines_dir = "engines"
engine_name = "Pikafish"
executable_name = "pikafish-bmi2.exe"
board_position_file = "board_position.txt"
ArmIK_dir = "ArmIK"
slideway = "slideway"

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)

# 构建要加载的文件的路径
checkerboard_12_pixels_position_path = os.path.join(current_dir, ArmIK_dir, "checkerboard_12_pixels_position.npy")
# print(checkerboard_12_pixels_position_path)
checkerboard_12_pulse_list_robot_trans_xy_path = os.path.join(current_dir, ArmIK_dir,
                                                              "checkerboard_12_pulse_list_robot_trans_xy.npy")
board_matrix_path = os.path.join(current_dir, "board_matrix.npy")
slideway_coordinates_path = os.path.join(current_dir, slideway, "slideway_coordinates.txt")

# 构建绝对路径
engine_address = os.path.abspath(os.path.join(base_dir, src_dir, engines_dir, engine_name, executable_name))
checkerboard_12_pulse_dict_file_path = os.path.join(current_dir, ArmIK_dir, "checkerboard_12_pulse_list_robot.txt")
board_position_file_path = os.path.join(current_dir, "board_position.txt")
roi_parameter_path = os.path.join(current_dir, "roi_parameter.txt")
hough_circles_parameter_path = os.path.join(current_dir, "hough_circles_parameter.txt")


try:
    board_matrix = np.load(board_matrix_path)
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("board_matrix不存在")

# 加载文件
try:
    checkerboard_12_pixels_position_array = np.load(checkerboard_12_pixels_position_path)
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("checkerboard_12_pixels_position_array不存在")

try:
    checkerboard_12_pulse_list_robot_trans_xy_array = np.load(checkerboard_12_pulse_list_robot_trans_xy_path)
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("checkerboard_12_pulse_list_robot_trans_xy_array不存在")


def save_checkerboard_12_pulse_list_robot_trans_xy_array(robot_position_list_x_y):
    robot_position_x_y_array = np.array(robot_position_list_x_y)
    np.save(checkerboard_12_pulse_list_robot_trans_xy_path, robot_position_x_y_array)
    print("robot_position_x_y_array保存成功")


try:
    with open(board_position_file_path, 'r', encoding='utf-8') as f:
        chess_board_dic = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("象棋位置文件不存在")


try:
    with open(checkerboard_12_pulse_dict_file_path, 'r', encoding='utf-8') as f:
        checkerboard_12_pulse_dict = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("checkerboard_12_pulse_dict不存在")


def save_checkerboard_12_pulse_dict(num_id, pulse_list):
    with open(checkerboard_12_pulse_dict_file_path, 'w', encoding='utf-8') as f:
        checkerboard_12_pulse_dict[num_id] = pulse_list
        f.write(json.dumps(checkerboard_12_pulse_dict))
        print('坐标{}保存完成!数值为{}'.format(num_id, pulse_list))


def save_roi_parameter(start_pos, last_pos):
    with open(roi_parameter_path, 'w', encoding='utf-8') as f:
        _roi_parameter_list = [start_pos, last_pos]
        f.write(json.dumps(_roi_parameter_list))
        print('roi保存完成!数值为{},{}'.format(start_pos, last_pos))


try:
    with open(roi_parameter_path, 'r', encoding='utf-8') as f:
        roi_parameter_list = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("roi文件不存在")


try:
    with open(hough_circles_parameter_path, 'r', encoding='utf-8') as f:
        hough_circles_parameter_dict = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("霍尔圆文件参数不存在")


# 标准棋盘列表
standard_chess_board = chess_board_dic["chess_board"]


try:
    with open(slideway_coordinates_path, 'r', encoding='utf-8') as f:
        slideway_coordinates_dict = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("roi文件不存在")


class SaveLoadPicture:
    def __init__(self):
        self.output_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'output')
        self.output_after_board_dir = os.path.join(self.output_dir, 'after_board')
        self.output_before_board_dir = os.path.join(self.output_dir, 'before_board')

    def save_before_board(self, name, image, suffix=".png"):
        image = np.array(image).copy()
        file_path = os.path.join(self.output_before_board_dir, str(name) + suffix)
        cv2.imwrite(file_path, image)

    def save_after_board(self, name, image, suffix=".png"):
        image = np.array(image).copy()
        file_path = os.path.join(self.output_after_board_dir, str(name) + suffix)
        cv2.imwrite(file_path, image)

    def load_before_board(self, name, suffix=".png"):
        file_path = os.path.join(self.output_before_board_dir, str(name) + suffix)
        picture = cv2.imread(file_path)
        return picture

    def load_after_board(self, name, suffix=".png"):
        file_path = os.path.join(self.output_after_board_dir, str(name) + suffix)
        picture = cv2.imread(file_path)
        return picture

    def save_picture(self, name, image, suffix=".png"):
        image = np.array(image).copy()
        file_path = os.path.join(self.output_dir, str(name) + suffix)
        cv2.imwrite(file_path, image)

    def load_picture(self, name, suffix=".png"):
        file_path = os.path.join(self.output_dir, str(name) + suffix)
        picture = cv2.imread(file_path)
        return picture

    # def save_save_multiple_frame_results(self, frame=25):

def set_x_y_pixel_limit(x_pixel, y_pixel):
    if x_pixel <= roi_parameter_list[0][0] or x_pixel >= roi_parameter_list[1][0]:
        return False
    if y_pixel <= roi_parameter_list[0][1] or y_pixel >= roi_parameter_list[1][1]:
        return False
    return True


if __name__ == '__main__':
    Picture = SaveLoadPicture()
    print(checkerboard_12_pixels_position_array[0])
    print(checkerboard_12_pulse_dict["0"])
    # print(ROOT)
    # print(path)
