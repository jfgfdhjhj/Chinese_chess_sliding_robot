#!/usr/bin/env python3
# encoding:utf-8
import cv2
import numpy as np
import math
import json
from chess_utils.logger import logger
from system import checkerboard_12_pixels_position_array, roi_parameter_list
# 测试图片保存路径
TestPicturePath = 'TestPicture/'
OutPicturePath = "output/"
file = "board_position.txt"
checkerboard_parameter_path = "checkerboard_parameter.txt"

# # 单个棋盘宽度，2.5 * 8单位cm
# weight = 2.5
# # 单个棋盘高度，2. 64 * 9单位cm
# height = 2.64


cap = cv2.VideoCapture(1)
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 1920, 1080)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

FLAG_MOUSE_CLIK = 0
start_pos = (0, 0)
finish_pos = (0, 0)

weight = roi_parameter_list[1][0] - roi_parameter_list[0][0]
high = roi_parameter_list[1][1] - roi_parameter_list[0][1]

# 棋盘有没有圆圈，有为Ture，没有改为False
is_test = False
# 找到之后这里改成True，没找到改成False
is_find_4_corner = True

if is_find_4_corner:
    try:
        with open(checkerboard_parameter_path, 'r', encoding='utf-8') as f:
            _checkerboard_parameter_dict = json.loads(f.read())
    except FileNotFoundError:  # 抛出文件不存在异常
        logger.warning("棋盘位置参数文件不存在")


# 矩形角度最小阈值回调函数
def rectangle_minimum_threshold(value):
    # print("value")
    pass


# 矩形角度最大阈值回调函数
def rectangle_maximum_threshold(value):
    # print("value")
    pass


def calculate_angle3(p1, p2, p3):
    # 计算向量 P1P2 和 P2P3
    vec1 = [p1[0] - p2[0], p1[1] - p2[1]]
    vec2 = [p3[0] - p2[0], p3[1] - p2[1]]

    # 计算点积
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

    # 计算向量长度
    length_vec1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    length_vec2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    # 计算夹角的余弦值
    cos_angle = dot_product / (length_vec1 * length_vec2)

    # 计算角度
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def find_rectangle_corners(hull1, angle_threshold1):
    corners = []
    for i in range(len(hull1)):
        point1 = tuple(hull1[i % len(hull1)][0])
        point2 = tuple(hull1[(i + 1) % len(hull1)][0])
        point3 = tuple(hull1[(i + 2) % len(hull1)][0])
        angle = calculate_angle3(point1, point2, point3)
        if angle_threshold1[0] <= angle <= angle_threshold1[1]:
            corners.append(point2)

    return corners


def chess_board_sort(points_all=None, delta=2):
    """
    接受几个矩形坐标，进行计算和排序
    :param points_all:
    :param delta: 相邻坐标的误差
    :return: 返回计算后的四个坐标，从左上角开始逆时针排序排列返回列表
    """
    global is_test
    if not is_test:
        if len(points_all) < 4:
            logger.error("输入的参数小于4个，请检查")
            return 0
        logger.debug("输入的坐标列表{}".format(points_all))
        corners = []
        corners_left = []
        corners_right = []
        # 提取每个元组的横坐标
        x_coordinates = [point[0] for point in points_all]

        # 计算横坐标的平均值
        average_x = sum(x_coordinates) / len(x_coordinates)
        for point in points_all:
            # 这里需要估计左边两个点大致的x轴的坐标，也就是cv2坐标的第一个点，要不然会报错
            if 0 <= point[0] <= average_x:
                corners_left.append(point)
            else:
                corners_right.append(point)
        # 找到最靠左
        left_top_most = min(corners_left, key=lambda x: x[1])
        left_bottom_most = max(corners_left, key=lambda x: x[1])
        # 找到最靠右
        right_top_most = min(corners_right, key=lambda x: x[1])
        right_bottom_most = max(corners_right, key=lambda x: x[1])

        corners.extend([left_top_most, right_top_most, right_bottom_most, left_bottom_most])
        logger.debug("得出的坐标：{}".format(corners))
        if (len(corners)) < 4:
            logger.error("参数小于4个，错误")
            return 0
        # # 计算四个点的重心（矩心）
        #
        # center_x = sum(point[0] for point in corners) / 4
        # center_y = sum(point[1] for point in corners) / 4
        # center = (center_x, center_y)
        # # 根据重心的坐标，将四个点分为两组
        # top_points = [point for point in corners if point[1] < center[1]]
        # bottom_points = [point for point in corners if point[1] >= center[1]]
        #
        # # 对上面两个点按照 x 坐标进行排序
        # top_points.sort(key=lambda p: p[0])
        #
        # # 对下面两个点按照 x 坐标进行排序
        # bottom_points.sort(key=lambda p: p[0])
        #
        # # 左上角、右上角、右下角、左下角的顺序
        # sorted_points = [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]
        # print(sorted_points)
        return corners
    else:
        # 要提取的列的索引
        columns_to_extract = [0, 3, 11, 8]
        print(checkerboard_12_pixels_position_array)
        # 存储提取的坐标的列表
        coordinate_list = []
        for column_index in columns_to_extract:
            # 提取指定列的元素
            column_values = checkerboard_12_pixels_position_array[column_index, :]
            # 将提取的列添加到列表中
            coordinate_list.append(column_values)
        return coordinate_list


# 计算水平和垂直方向上的单位向量
def unit_vector(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]


def interpolate_points(top_left, top_right, bottom_left, bottom_right, num_rows=9, num_columns=10):

    # 水平和垂直方向上的网格数
    horizontal_grid_count = num_columns - 1
    vertical_grid_count = num_rows - 1

    # 计算水平和垂直方向上的单位向量
    horizontal_unit_vector = unit_vector(top_left, top_right)
    vertical_unit_vector = unit_vector(top_left, bottom_left)

    # 计算水平和垂直方向上的增量向量
    horizontal_increment = (horizontal_unit_vector[0] / horizontal_grid_count,
                            horizontal_unit_vector[1] / horizontal_grid_count)
    vertical_increment = (vertical_unit_vector[0] / vertical_grid_count, vertical_unit_vector[1] / vertical_grid_count)

    # 计算其他点的坐标
    points1 = []
    i = 0
    for row in range(num_rows):
        j = 0
        for column in range(num_columns):
            x = int(top_left[0] + column * horizontal_increment[0] + row * vertical_increment[0])
            y = int(top_left[1] + column * horizontal_increment[1] + row * vertical_increment[1])
            points1.append((i, j, x, y))
            j += 1
        i += 1
    dxi = int(abs((top_right[0] - top_left[0])/(num_columns - 1)))
    dyi = int(abs((top_left[1] - bottom_left[1])/(num_rows - 1)))
    return points1, dxi, dyi


def calculate_chessboard_coordinates(start_x, start_y, cell_width, cell_height, num_columns=10, num_rows=9):
    """
    :param start_x: 左上角 x 坐标
    :param start_y: 左上角 y 坐标
    :param cell_width: 格子宽度
    :param cell_height: 格子高度
    :param num_columns: 列数
    :param num_rows: 行数
    :return:
    """
    coordinates = []
    for col in range(num_columns):
        for row in range(num_rows):
            x = start_x + col * cell_width
            y = start_y + row * cell_height
            coordinates.append((col, row, x, y))
    return coordinates


# 创建矩形最小角度阈值滑动条
cv2.createTrackbar('angleMin', 'window', -360, 360, rectangle_minimum_threshold)
# 设置滑动条的默认值为133
cv2.setTrackbarPos('angleMin', 'window', 133)
# 创建矩形最大角度阈值滑动条
cv2.createTrackbar('angleMax', 'window', -360, 360, rectangle_maximum_threshold)
# 设置滑动条的默认值为133
cv2.setTrackbarPos('angleMax', 'window', 151)

while cap.isOpened() and 1:
    flag, image = cap.read()
    # start_pos = roi_parameter_list[0]
    # finish_pos = roi_parameter_list[1]
    # roi = image[roi_parameter_list[0][1]:roi_parameter_list[1][1], roi_parameter_list[0][0]:roi_parameter_list[1][0]]
    # image = roi
    if not flag:
        logger.error("没有打开摄像头，请检查")
        break
    elif not is_test:
        a = cv2.getTrackbarPos('angleMin', "window")
        b = cv2.getTrackbarPos('angleMax', "window")

        # 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化
        thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 去噪
        binary = cv2.medianBlur(binary, 5)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        external_contours = []
        max_area = 0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and area > max_area:
                max_area = area
                external_contours = [contour]

        for contour in external_contours:
            hull = cv2.convexHull(contour)

            # 在原始图像上绘制凸包
            cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
            # print(hull)

            if is_find_4_corner:
                # 找到矩形拐角,角度阈值范围
                angle_threshold = (_checkerboard_parameter_dict["angle_threshold"][0],
                                   _checkerboard_parameter_dict["angle_threshold"][1])
            else:
                # 角度阈值滑块范围
                angle_threshold = (a, b)

            rectangle_corners = find_rectangle_corners(hull, angle_threshold)
            logger.debug(f"rectangle_corners:{rectangle_corners}")
            for corner in rectangle_corners:
                cv2.circle(image, corner, 5, (0, 255, 255), -1)
            # 在图像上标记拐角
            if is_find_4_corner:
                rectangle_corners = chess_board_sort(rectangle_corners)
                if rectangle_corners == 0:
                    logger.error("参数错误，请调整")
                    break

                rectangle_corners = [tuple(int(x) for x in point) for point in rectangle_corners]

                # 左上角、右上角、右下角和左下角的坐标
                top_left = rectangle_corners[0]
                top_right = rectangle_corners[1]
                bottom_right = rectangle_corners[2]
                bottom_left = rectangle_corners[3]
                logger.debug("top_left:{}, top_right:{}, bottom_left:{}".format(top_left, top_right, bottom_left))
                _, dx, dy = interpolate_points(top_left, top_right, bottom_left, bottom_right)
                # 定义棋盘角点坐标
                board_contour = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

                start_pos_top_left_x = top_left[0]
                start_pos_top_left_y = top_left[1]

                target_corners = np.array([[start_pos_top_left_x, start_pos_top_left_y],
                                           [dx * 9, start_pos_top_left_y], [dx * 9, dy * 8],
                                           [start_pos_top_left_x, dy * 8]],
                                          dtype=np.float32)
                logger.debug("目标角点矩阵：{}".format(target_corners))
                points, _, _ = interpolate_points((start_pos_top_left_x, start_pos_top_left_y),
                                                  (dx * 9, start_pos_top_left_y),
                                                  (start_pos_top_left_x, dy * 8), (dx * 9, dy * 8))
                logger.debug("根据左上角点以及像素的长宽高计算出的所有点的理想坐标{}".format(points))
                # 计算透视变换矩阵
                m = cv2.getPerspectiveTransform(board_contour, target_corners)
                np.save("board_matrix.npy", m)
                logger.info("矩阵变换文件写入成功")

                # # 进行透视变换
                # board_corners = cv2.perspectiveTransform(np.array([board_contour]), m)
                # 设置透视变换矩阵的第三列
                # 进行透视变换
                dst_img = cv2.warpPerspective(image, m, (weight, high))

                # 计算逆透视变换矩阵
                m_inv = np.linalg.inv(m)
                _points = []
                for o1, o2, o3, o4 in points:
                    # 定义待逆变换的点坐标
                    target_point = np.array([o3, o4], dtype=np.float32)
                    # 进行逆透视变换
                    original_point = cv2.perspectiveTransform(target_point.reshape(-1, 1, 2), m_inv)
                    x, y = original_point[0, 0]
                    _original_point = (int(x), int(y))
                    _points.append((o1, o2, int(x), int(y)))
                    cv2.circle(image, _original_point, 5, (0, 0, 255), -1)
                logger.debug("变化的棋盘坐标：{}".format(_points))
                logger.debug("dx:{}, dy:{}".format(dx, dy))
                chess_board = {"chess_4position": rectangle_corners, "dx": dx, "dy": dy, "chess_board": _points}
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(chess_board))
                logger.info('配置文件写入完成!')

    else:
        if is_find_4_corner:
            rectangle_corners = chess_board_sort()
            if rectangle_corners == 0:
                logger.error("参数错误，请调整")
                break
            rectangle_corners = [tuple(int(x) for x in point) for point in rectangle_corners]
            # 左上角、右上角、右下角和左下角的坐标
            top_left = rectangle_corners[0]
            top_right = rectangle_corners[1]
            bottom_right = rectangle_corners[2]
            bottom_left = rectangle_corners[3]
            logger.debug("top_left:{}, top_right:{}, bottom_left:{}".format(top_left, top_right, bottom_left))
            _, dx, dy = interpolate_points(top_left, top_right, bottom_left, bottom_right)
            # 定义棋盘角点坐标
            board_contour = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

            start_pos_top_left_x = top_left[0]
            start_pos_top_left_y = top_left[1]

            target_corners = np.array([[start_pos_top_left_x, start_pos_top_left_y],
                                       [dx * 9, start_pos_top_left_y], [dx * 9, dy * 8],
                                       [start_pos_top_left_x, dy * 8]], dtype=np.float32)
            print(target_corners)
            points, _, _ = interpolate_points((start_pos_top_left_x, start_pos_top_left_y),
                                              (dx * 9, start_pos_top_left_y),
                                              (start_pos_top_left_x, dy * 8), (dx * 9, dy * 8))
            print(points)
            # 计算透视变换矩阵
            m = cv2.getPerspectiveTransform(board_contour, target_corners)
            np.save("board_matrix.npy", m)
            logger.info("矩阵变换文件写入成功")
            # # 进行透视变换
            # board_corners = cv2.perspectiveTransform(np.array([board_contour]), m)
            # 设置透视变换矩阵的第三列
            print(m)
            # 进行透视变换
            dst_img = cv2.warpPerspective(image, m, (1920, 1080))  # 设置输出图像的大小为 (1920, 1080)
            # 计算逆透视变换矩阵
            m_inv = np.linalg.inv(m)
            _points = []
            for o1, o2, o3, o4 in points:
                # 定义待逆变换的点坐标
                target_point = np.array([o3, o4], dtype=np.float32)
                # 进行逆透视变换
                original_point = cv2.perspectiveTransform(target_point.reshape(-1, 1, 2), m_inv)
                x, y = original_point[0, 0]
                _original_point = (int(x), int(y))
                _points.append((o1, o2, int(x), int(y)))
                cv2.circle(image, _original_point, 5, (0, 0, 255), -1)
            logger.debug("变化的棋盘坐标：{}".format(_points))
            logger.debug("dx:{}".format(dx))
            logger.debug("dy:{}".format(dy))
            chess_board = {"chess_4position": rectangle_corners, "dx": dx, "dy": dy, "chess_board": _points}
            with open(file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(chess_board))
            logger.info('配置文件写入完成!')
        # cv2.imwrite(OutPicturePath + "board.jpg", image)
    if is_find_4_corner:
        time = 0
    else:
        time = 100
    if is_find_4_corner:
        if is_test:
            # 显示透视后的图像
            cv2.imshow('Perspective Transformed Image', dst_img)
            cv2.imshow('window', image)
        else:
            cv2.imshow('Perspective Transformed Image', dst_img)
            # cv2.resizeWindow("window1", roi_parameter_list[1][0] - roi_parameter_list[0][0],
            #                  roi_parameter_list[1][1] - roi_parameter_list[0][1])
            cv2.imshow('window_change', image)
    else:
        cv2.imshow('window', image)
    key = cv2.waitKey(time)

    if key & 0xFF == ord('q'):
        # 将NumPy数组转换为Python列表
        if not is_find_4_corner:
            checkerboard_parameter_dict = {}
            checkerboard_parameter_dict["angle_threshold"] = [a, b]
            # 将每个元组中的整数转换为整数列表
            formatted_corners = [[int(coord) for coord in corner] for corner in rectangle_corners]
            checkerboard_parameter_dict["angular_coordinates"] = formatted_corners
            print(rectangle_corners)
            with open(checkerboard_parameter_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(checkerboard_parameter_dict))
            logger.info('边角位置,阈值配置文件写入完成!')
        # print(external_contours)
        cv2.destroyAllWindows()
        cap.release()
        break
