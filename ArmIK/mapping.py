import cv2
import numpy as np
from matplotlib import pyplot as plt

from ArmIK.modified_kinematics import pulse_trans_theta, RobotKinematics, theta_trans_pulse
from system import checkerboard_12_pulse_dict, board_matrix
import os
from chess_utils.logger import logger
from system import checkerboard_12_pixels_position_array
# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
affine_matrix_path = os.path.join(current_dir, "affine_matrix")
checkerboard_12_pulse_list_robot_trans_xy_path = os.path.join(current_dir,
                                                              "checkerboard_12_pulse_list_robot_trans_xy.npy")
# 棋盘宽2.5cm
dar_x = 2.5
# 棋盘长
dar_y = 2.64


class HandInEyeCalibration:
    def __init__(self):
        # 通过12点标定获取的圆心相机坐标,从左上角开始，从左到右，分为三行

        self.STC_points_robot_fit = None
        self.fit_func2 = None
        self.fit_func1 = None
        self.STC_points_camera = checkerboard_12_pixels_position_array
        print(self.STC_points_camera)
        robot_position_list = []
        for i_list in checkerboard_12_pulse_dict.keys():
            ik = RobotKinematics()
            theta1, theta2, theta3, theta4, theta5, theta6 = (
                pulse_trans_theta(checkerboard_12_pulse_dict[i_list]))
            robot_position = ik.forward_kinematics(theta1, theta2, theta3, theta4)
            robot_position_list.append((robot_position[1], robot_position[2]))
        # self.robot_position_list = robot_position_list
        robot_position_array = np.array(robot_position_list)
        self.robot_position_array = robot_position_array
        print(robot_position_array)
        np.save(checkerboard_12_pulse_list_robot_trans_xy_path, robot_position_array)

        self.STC_points_robot = np.array([[pos[0], pos[1]] for pos in robot_position_list])
        self._save_affine_matrix()
        self.limit_y = (robot_position_list[4][1] +
                        robot_position_list[5][1] +
                        robot_position_list[6][1] +
                        robot_position_list[7][1])/4
        # self.cur_fit()
        points_camera_trans_affine_point = cv2.perspectiveTransform(checkerboard_12_pixels_position_array.reshape(-1, 1, 2), board_matrix)
        self.points_camera_trans_affine_point = points_camera_trans_affine_point.reshape(-1, 2).astype(int)
        logger.info("points_camera_trans_affine_point:{}".format(self.points_camera_trans_affine_point))

    def get_m(self, points_camera, points_robot):
        """
        取得相机坐标转换到机器坐标的仿射矩阵
        :param points_camera:
        :param points_robot:
        :return:
        """
        # 确保两个点集的数量级不要差距过大，否则会输出None
        m, _ = cv2.estimateAffine2D(points_camera, points_robot)

        return m

    def get_points_robot(self, x_camera, y_camera):
        """
        相机坐标通过仿射矩阵变换取得机器坐标
        :param x_camera:
        :param y_camera:
        :return:
        """
        # m = self.get_m(self.STC_points_camera, self.STC_points_robot)
        x_y_array = np.array([x_camera, y_camera], dtype=np.float32).reshape(-1, 1, 2)
        output_camera_trans_affine_point = cv2.perspectiveTransform(x_y_array, board_matrix)
        output_camera_trans_affine_point = output_camera_trans_affine_point.reshape(-1).astype(int)
        print("output_camera_trans_affine_point:{}".format(output_camera_trans_affine_point))
        m = self.get_m(self.points_camera_trans_affine_point, self.STC_points_robot)
        # robot_x = round((m[0][0] * x_camera) + (m[0][1] * y_camera) + m[0][2], 2)
        robot_x = round((m[0][0] * output_camera_trans_affine_point[0]) + (m[0][1] * output_camera_trans_affine_point[1]) + m[0][2], 2)
        # robot_y = round((m[1][0] * x_camera) + (m[1][1] * y_camera) + m[1][2], 2)
        robot_y = round((m[1][0] * output_camera_trans_affine_point[0]) + (m[1][1] * output_camera_trans_affine_point[1]) + m[1][2], 2)

        # # 相机坐标系中的点
        # point_camera = np.array([[x_camera, y_camera]])
        #
        # # 应用仿射变换矩阵，将相机坐标系中的点转换为机器人坐标系中的点
        # point_robot = cv2.transform(np.array([point_camera]), m)
        # print(point_robot)
        logger.debug("映射出的机器人坐标为：robot_x:{}, robot_y:{}".format(robot_x, robot_y))
        return robot_x, robot_y

    def get_points_robot_test(self, x_camera, y_camera):
        """
        相机坐标通过仿射矩阵变换取得机器坐标
        :param x_camera:
        :param y_camera:
        :return:
        """
        robot_x = ((self.affine_matrix[0][0] * x_camera) + (self.affine_matrix[0][1] * y_camera) +
                   self.affine_matrix[0][2])
        robot_y = ((self.affine_matrix[1][0] * x_camera) + (self.affine_matrix[1][1] * y_camera) +
                   self.affine_matrix[1][2])
        return robot_x, robot_y

    def _save_affine_matrix(self):
        m = self.get_m(self.STC_points_camera, self.STC_points_robot)
        # 保存矩阵到 .npz 文件
        np.save(affine_matrix_path, m)
        self.affine_matrix = m
        print(m)
        print("仿射矩阵已保存到文件:", affine_matrix_path)

    def limit_theta4_according_to_robot_coordinates(self, robot_y):
        if robot_y <= self.limit_y:
            _theta4 = 90
            return _theta4
        else:
            _theta4 = 48.24
            return _theta4

    def cur_fit(self):
        # 拟合第一组数据
        x1 = self.robot_position_array[:, 0]
        _list = []
        for pixel_x, pixel_y in self.STC_points_camera:
            _x, _y = self.get_points_robot(pixel_x, pixel_y)
            # print(_x, _y)
            _list.append((_x, _y))
        array = np.array(_list)
        self.STC_points_robot_fit = array
        # print(array)
        y1 = array[:, 0]
        degree1 = 2  # 多项式次数
        coefficients1 = np.polyfit(x1, y1, degree1)
        self.fit_func1 = np.poly1d(coefficients1)

        # 拟合第二组数据
        x2 = self.robot_position_array[:, 1]
        y2 = array[:, 1]
        degree2 = 4  # 多项式次数
        coefficients2 = np.polyfit(x2, y2, degree2)
        self.fit_func2 = np.poly1d(coefficients2)

        # 计算拟合值
        y1_fit = self.fit_func1(x1)
        y2_fit = self.fit_func2(x2)

        # 绘制拟合曲线和数据点
        plt.subplot(1, 2, 1)
        plt.plot(x1, y1, 'o', label='Data 1')
        plt.plot(x1, y1_fit, label='Fit 1')
        plt.xlabel('X_pixels')
        plt.ylabel('x_robot')
        plt.title('Data 1 Polynomial Fitting')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(x2, y2, 'o', label='Data 2')
        plt.plot(x2, y2_fit, label='Fit 2')
        plt.xlabel('Y_pixels')
        plt.ylabel('y_robot')
        plt.title('Data 2 Polynomial Fitting')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def cal_fit(self, x_robot, y_robot):
        # 计算拟合值
        y1_fit = self.fit_func1(x_robot)
        y2_fit = self.fit_func2(y_robot)

        return y1_fit, y2_fit

    def coordinate_mapping_trans_my_pulse(self, pixel_x, pixel_y, z=5):
        # _x, _y = self.cal_fit(pixel_x, pixel_y)
        _x, _y = self.get_points_robot(pixel_x, pixel_y)
        # _x, _y = self.cal_fit(_x, _y)
        # print("2:{},{}".format(_x, _y))
        alpha_j4 = self.limit_theta4_according_to_robot_coordinates(_y)
        _ik = RobotKinematics()
        theta_list = _ik.inverse_kinematics_limit_theta4(_x, _y, z, alpha_j4=alpha_j4)
        logger.info("根据图片映射出的机械臂角度列表为：{}".format(theta_list))
        my_pwm_list = theta_trans_pulse(theta_list)[0]
        logger.info("机械臂的脉冲列表为，序号为机械臂底部开始命名{}".format(my_pwm_list))

        return my_pwm_list


if __name__ == "__main__":
    hand = HandInEyeCalibration()
    hand.coordinate_mapping_trans_my_pulse(343, 188)
    hand.coordinate_mapping_trans_my_pulse(1329, 150)
    hand.coordinate_mapping_trans_my_pulse(1362, 1010)
