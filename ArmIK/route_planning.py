import re
import time
import ArmIK.PID as PID
import ArmIK.mapping as mapping
from socket_client import Client, ClientEle
from chess_utils.logger import logger
import system
# from system import checkerboard_12_pulse_dict

from ArmIK.modified_kinematics import (pulse_trans_theta, RobotKinematics,
                                       theta_trans_pulse)


class Route:
    Calibration_coordinate_PATTERN = 1
    move_time = 1000

    def __init__(self):
        self.pid = PID.PID(P=0.01, I=0.001, D=0)
        self.hand = mapping.HandInEyeCalibration()
        self.clint = Client(_is_vm=False)
        self.clint_ele = ClientEle(_is_vm=False)

    def pid_adjust(self, servo_id, target_pulse, err=5, times=10):
        self.pid.SetPoint = target_pulse
        for i in range(times):
            current_pulse = self.clint.send_get_pulse(state_id=servo_id)
            logger.debug("id{},current_pulse{}".format(servo_id, current_pulse))
            self.pid.update(current_pulse)
            pulse_change = int(self.pid.output)
            c_last_pulse = int(current_pulse + pulse_change)
            self.clint.send_move((int(servo_id), c_last_pulse, 20))
            last_pulse5 = self.clint.send_get_pulse(state_id=servo_id)
            if abs(target_pulse - last_pulse5) < err:
                logger.info("current_pulse:{},小于{},pid调参成功".format(current_pulse, err))
                return True
        return False

    def arrival_pulse(self, pulse_list=None, is_checkerboard_id=True, checkerboard_id="0", is_leave_gaps=True,
                      lifting_height=50):
        """

        """
        if is_checkerboard_id:
            target_pulse5 = system.checkerboard_12_pulse_dict[checkerboard_id][1]
            pulse_list1 = system.checkerboard_12_pulse_dict[checkerboard_id][:]
        else:
            target_pulse5 = pulse_list[1]
            pulse_list1 = pulse_list[:]

        # 第一步，移动到接近点
        pulse_list1[1] = pulse_list1[1] + 50
        self.clint.send_move((6, pulse_list1[0], 1500), (5, pulse_list1[1], 1500), (4, pulse_list1[2], 1500),
                             (3, pulse_list1[3], 1500), (2, pulse_list1[4], 1500), (1, pulse_list1[5], 1500))
        self.clint.send_move((5, target_pulse5, 1000))
        # self.clint.send_move((5, target_pulse5 + 9, 1000))

        # 第二步，调整id5
        if not is_leave_gaps:
            pid_res = self.pid_adjust(servo_id=5, target_pulse=target_pulse5, err=3, times=2)
            if not pid_res:
                self.clint.send_move((5, target_pulse5, 1000))
        # 第三部，抬起
        current_pulse_5 = self.clint.send_get_pulse(state_id=5)
        self.clint.send_move((5, current_pulse_5 + lifting_height, 1000))

    def move_one_point_to_another_point(self, checkerboard_pixel_before, checkerboard_pixel_after, is_eat=True):
        checkerboard_before_pulse = self.hand.coordinate_mapping_trans_my_pulse(checkerboard_pixel_before[0],
                                                                                checkerboard_pixel_before[1], z=3)
        checkerboard_after_pulse = self.hand.coordinate_mapping_trans_my_pulse(checkerboard_pixel_after[0],
                                                                               checkerboard_pixel_after[1], z=4)
        # 第一步, 机械臂位置归位：
        self.clint.send_init()
        if is_eat:
            self.clint_ele.send_electromagnet_attract()
            self.arrival_pulse(checkerboard_after_pulse, is_checkerboard_id=False, is_leave_gaps=True, lifting_height=50)
            time.sleep(1)
            self.clint.send_init()
            self.clint_ele.send_electromagnet_fall()
            self.clint_ele.send_electromagnet_attract()
            self.arrival_pulse(checkerboard_before_pulse, is_checkerboard_id=False, is_leave_gaps=True)
            time.sleep(1)
            self.arrival_pulse(checkerboard_after_pulse, is_checkerboard_id=False, is_leave_gaps=True)
            self.clint_ele.send_electromagnet_fall()
            self.clint.send_init()
        else:
            self.clint_ele.send_electromagnet_attract()
            self.arrival_pulse(checkerboard_before_pulse, is_checkerboard_id=False, is_leave_gaps=True, lifting_height=50)
            time.sleep(1)
            self.arrival_pulse(checkerboard_after_pulse, is_checkerboard_id=False, is_leave_gaps=True, lifting_height=10)
            self.clint_ele.send_electromagnet_fall()
            self.clint.send_init()


if __name__ == '__main__':
    import cv2
    from chess_trans import recognition_circle_multiple, RobotMovePlanning, fen_move_trans_key_id
    from system import standard_chess_board
    robot_image_before = recognition_circle_multiple(time=5)[1]

    # robot_recognize = RobotMovePlanning(robot_image_before, data[1], res1[1])
    imgSource = robot_image_before.copy()
    # 转为灰度图
    gray = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)

    # 中值滤波
    imgMedian = cv2.medianBlur(gray, 3)

    # 直方图均衡化
    imgEqualizeHist = cv2.equalizeHist(imgMedian)

    # # 二值化
    # thresh, threshold_two = cv2.threshold(imgEqualizeHist, 100, 255, cv2.THRESH_BINARY)

    # 输入图像，方法（类型），dp(dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推)，
    # 最短距离-可以分辨是两个圆否 则认为是同心圆 ,边缘检测时使用Canny算子的高阈值，中心点累加器阈值—候选圆心（霍夫空间内累加和大于该阈值的点就对应于圆心），检测到圆的最小半径，检测到圆的的最大半径
    # 用a代替param1, 用c代替param2 # 27

    circles = cv2.HoughCircles(imgEqualizeHist, cv2.HOUGH_GRADIENT, dp=1, minDist=38, param1=91, param2=19,
                               minRadius=27, maxRadius=32)
    route = Route()
    # 检测到了圆
    if circles is not None:
        # 遍历每一个圆
        for circle in circles[0]:
            x, y, r1 = int(circle[0]), int(circle[1]), int(circle[2])
            print(x, y)
            fen_move = key = input("请输入点:")
            key_id = fen_move_trans_key_id(fen_move)
            print(key_id)
            after_pixel = standard_chess_board[key_id[1]]
            print(after_pixel)
            print(after_pixel[2], after_pixel[3])
            route.move_one_point_to_another_point([x, y], [after_pixel[2], after_pixel[3]], is_eat=False)



    # route = Route()
    # ik = RobotKinematics()
    # # route.move_one_point_to_another_point([341, 185], [1329, 150], is_eat=False)
    # # route.move_one_point_to_another_point([343, 173], [1332, 138], is_eat=False)
    # # route.arrival_pulse(pulse_list=None, is_checkerboard_id=True, checkerboard_id="0", is_leave_gaps=True)
    # while True:
    #     # 信息发送
    #     info = input('请输入命令，p获取pulse,h末端垂直,w微调,q退出,t坐标测试,d掉电, a吸取, f掉落,保存坐标按s:')
    #     if info == "p":
    #         decoded_list = route.clint.send_get_pulse()
    #         print("decoded_list:{}".format(decoded_list))
    #         theta1, theta2, theta3, theta4, theta5, theta6 = pulse_trans_theta(decoded_list)
    #         print(theta1, theta2, theta3, theta4, theta5, theta6)
    #         try:
    #             _, x1, y1, z1 = ik.forward_kinematics(theta1, theta2, theta3, theta4)
    #             print("x1:{}, y1:{}, z1:{},".format(x1, y1, z1))
    #             alpha_j4 = route.hand.limit_theta4_according_to_robot_coordinates(y1)
    #             res = ik.inverse_kinematics_limit_theta4(x1, y1, z1, alpha_j4=alpha_j4)
    #             print("res:{}".format(res))
    #             _pwm_list = theta_trans_pulse(res)[0]
    #             print("推导出的逆运动学脉冲值列表为:{}".format(_pwm_list))
    #         except ValueError:
    #             logger.error("逆运动学解算失败！")
    #     elif info == "a":
    #         route.clint_ele.send_electromagnet_attract()
    #     elif info == "f":
    #         route.clint_ele.send_electromagnet_fall()
    #     elif info == "d":
    #         route.clint.send_power_down()
    #     elif info == "h":
    #         route.clint.send_vertical_end()
    #     elif info == "s":
    #         num_id = input("输入坐标：比如0到11:")
    #         system.save_checkerboard_12_pulse_dict(num_id, route.clint.send_get_pulse())
    #     elif info == "w":
    #         while True:
    #             key = input("请输入微调数值，比如6+2,h为末端垂直，退出按q：")
    #             if key == "q":
    #                 break
    #             elif key == "h":
    #                 route.clint.send_vertical_end()
    #             else:
    #                 matches = re.match(r"([-+]?\d+)([-+])(\d+)", key)
    #                 if matches:
    #                     servo_id = int(matches.group(1))
    #                     symbol = matches.group(2)
    #                     darta_pulse = int(matches.group(3))
    #                     if symbol == '-':
    #                         darta_pulse = -darta_pulse
    #                     route.clint.send_fine_adjustment((servo_id, darta_pulse))
    #                     print("servo_id:", servo_id)
    #                     print("Symbol:", symbol)
    #                     print("Number 2:", darta_pulse)
    #                 else:
    #                     print("No match found.")
    #     elif info == "t":
    #         while True:
    #             key = input("请输入坐标,范围0到11,h为末端垂直，微调按w,保存按s,z调整j4角度, 退出按q：")
    #             if key == "q":
    #                 break
    #             elif key == "z":
    #                 j4_key = input("现在将只调整j4角度，请确认j4位置不会碰撞, 90度按1, 另一个角度按2,")
    #                 if j4_key == "1":
    #                     route.clint.send_move((3, 125, 1000))
    #                 elif j4_key == "2":
    #                     route.clint.send_move((3, 300, 1000))
    #             elif key == "h":
    #                 route.clint.send_vertical_end()
    #             elif key in [str(num_id) for num_id in range(0, 12)]:
    #                 route.arrival_pulse(pulse_list=None, is_checkerboard_id=True, checkerboard_id=key)
    #             elif key == "s":
    #                 num_id = input("输入坐标：比如0到11")
    #                 system.save_checkerboard_12_pulse_dict(num_id, route.clint.send_get_pulse())
    #
    #             elif key == "w":
    #                 while True:
    #                     key = input("请输入微调数值，比如6+2,h为末端垂直，退出按q：,保存按s")
    #                     if key == "q":
    #                         break
    #                     elif key == "h":
    #                         route.clint.send_vertical_end()
    #                     elif key == "s":
    #                         num_id = input("输入坐标：比如0到11")
    #                         system.save_checkerboard_12_pulse_dict(num_id, route.clint.send_get_pulse())
    #                     else:
    #                         matches = re.match(r"([-+]?\d+)([-+])(\d+)", key)
    #                         if matches:
    #                             servo_id = int(matches.group(1))
    #                             symbol = matches.group(2)
    #                             darta_pulse = int(matches.group(3))
    #                             if symbol == '-':
    #                                 darta_pulse = -darta_pulse
    #                             route.clint.send_fine_adjustment((servo_id, darta_pulse))
    #                             print("servo_id:", servo_id)
    #                             print("Symbol:", symbol)
    #                             print("Number 2:", darta_pulse)
    #                         else:
    #                             print("No match found.")
    #
    #     elif info == "q":
    #         break
