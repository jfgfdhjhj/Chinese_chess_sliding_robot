import logging
import math
from chess_utils.logger import logger
from scipy.optimize import newton


class RobotKinematics:
    P = 0  # 8.5
    #  P为机械臂中心到原点的长度
    A1 = 7.05  # 6.5  # 6.1
    A2 = 10.16  # 10.15
    A3 = 9.64
    # A4 = 16.931  # 开 518，合住 624 16.331 - 11.2 + 0.4 + 0.4 + 5.6 + 5.4  = 16.931
    A4 = 16.95  # 16.331 - 11.2 + 0.4 + 0.4 + 5.6 + 5.0= 17.522  # 16.531

    A5 = 5.43  # 3.009 +  0.55 +  1 + 0.4 # 定义为末端吸铁石 # 3.959

    magnet_dx = 0.5
    magnet_dy = 0.5

    def __init__(self):
        self.rad = None
        self.rad_dict = None
        self.valid = None
        self.__MAX_LEN = self.A2 + self.A3 + self.A4
        self.__MAX_HIGH = self.A1 + self.A2 + self.A3 + self.A4

    def __valid_j(self, joint, j):
        """
        :param joint: 定义j1为底座，j2为底座往上，以此类推
        :param j:
        :return:
        """
        if j is None:
            self.valid = False
            return False
        degree = math.degrees(j)
        if joint == 1:
            if 0 <= degree <= 180:
                return True
        elif joint == 2:
            if -80 < degree < 90:
                return True
        elif joint == 3:
            if -5 <= degree < 110:
                return True
        elif joint == 4:
            if 0 <= degree <= 90:
                return True
        logger.warning('joint {} is invalid j:{} degree {}'.format(joint, j, degree))
        return False

    def __valid_degree(self, joint, degree):
        if 0 <= degree <= 180:
            return True
        else:
            logger.warning('joint {} is invalid degree {}'.format(joint, degree))
            self.valid = False
            return False

    def __out_of_range(self, length, height):
        if height > self.__MAX_HIGH:
            logger.warning('高度 {} 超过界限 {}'.format(height, self.__MAX_HIGH))
            return True
        if length > self.__MAX_LEN:
            logger.warning('投影长度 {} 超过界限 {}'.format(length, self.__MAX_LEN))
            return True
        return False

    def forward_kinematics(self, theta1, theta2, theta3, theta4):
        self.valid = None
        rad1, rad2, rad3, rad4 = math.radians(theta1), math.radians(theta2), math.radians(theta3), math.radians(theta4),
        length = self.A2 * math.sin(rad2) + self.A3 * math.sin(rad2 + rad3) + self.A4 * math.sin(rad2 + rad3 + rad4)
        height = self.A1 + self.A2 * math.cos(rad2) + self.A3 * math.cos(rad2 + rad3) + self.A4 * math.cos(rad2 + rad3 + rad4)
        alpha = theta2 + theta3 + theta4
        z = round(height - self.A5, 2)
        x = round(length * math.cos(rad1), 2)
        y = round(length * math.sin(rad1) - self.P, 2)
        # 世界坐标的边界
        if 0 <= y and z >= 0:
            self.valid = True

        x = x + self.magnet_dx
        y = y + self.magnet_dy
        logger.info('valid:{},x:{},y:{},z:{},lenghth:{},height:{},alpha:{}'.format(self.valid, x, y, z,
                                                                                    round(length, 2),
                                                                                    round(height, 2), alpha))

        return self.valid, x, y, z

    @staticmethod
    def alpha_sort_out(_rad_dict, max_alpha=180, min_alpha=130):
        alpha_list = []
        _rad_dict_list = []
        for key in _rad_dict.keys():
            if min_alpha <= _rad_dict[key][4] <= max_alpha:
                alpha_list.append(key)
        if len(alpha_list) > 1:
            logger.warning("筛选的角度范围满足的大于一个,目前的策略是取最大角度")
            max_value = max(alpha_list)
            _alpha = max_value
        else:
            _alpha = alpha_list

        return _rad_dict[_alpha]

    def inverse_kinematics(self, x, y, z, alpha=270, max_alpha=180, min_alpha=130, is_return_one=True, j4_theta=101):
        x = float(x)
        y = float(y)
        z = float(z)
        self.valid = False
        # rad2 = None
        # rad3 = None
        # rad4 = None
        rad_dict = {}
        # theta_list = []
        __MIN_ALPHA = 90  # j2+j3+j4 min value, 最后一个joint不向后仰
        num = 0
        __num_last = 100
        if z < 0:
            logger.warning('z 不能小于0')
            self.valid = False
            raise ValueError
        if y < 0:
            logger.warning('y 不能小于0')
            self.valid = False
            raise ValueError

        length = round(math.sqrt(pow((y + self.P), 2) + pow(x, 2)), 2)

        if length == 0:
            rad1 = 0  # 可以是任意数
        else:
            rad1 = math.atan2((y + self.P), x)

        height = z
        if self.__valid_j(1, rad1) and not self.__out_of_range(length, height) and not self.valid:
            while alpha >= __MIN_ALPHA:
                rad_alpha = math.radians(alpha)

                l0 = length - self.A4 * math.sin(rad_alpha)
                h0 = height - self.A4 * math.cos(rad_alpha) - self.A1
                logger.debug("l0:", l0, "h0:", h0)
                cos3 = (l0 ** 2 + h0 ** 2 - self.A2 ** 2 - self.A3 ** 2) / (2 * self.A2 * self.A3)
                if cos3 ** 2 > 1:
                    logger.debug("cos3大于1")
                    self.valid = False
                else:
                    rad3 = math.acos(cos3)
                    # sin3 = math.sqrt(1 - cos3 ** 2)
                    # rad3 = math.atan2(sin3, cos3)
                    # rad3 = math.radians(theta3)
                    if self.__valid_j(3, rad3):
                        k1 = self.A2 + self.A3 * cos3
                        k2 = self.A3 * math.sin(rad3)
                        logger.debug("K1:", k1, "K2", k2)
                        w = math.atan2(k2, k1)
                        logger.debug("w:", math.degrees(w))
                        logger.debug("theta3:", math.degrees(rad3))
                        rad2 = math.atan2(l0, h0) - w
                        if self.__valid_j(2, rad2):
                            rad4 = rad_alpha - rad3 - rad2
                            if self.__valid_j(4, rad4):
                                self.valid = True
                                theta_list = [round(math.degrees(rad1), 2), round(math.degrees(rad2), 2),
                                              round(math.degrees(rad3), 2), round(math.degrees(rad4), 2), alpha]
                                rad_dict[alpha] = theta_list
                                logger.debug(theta_list)

                if not self.valid:
                    alpha -= 1
                else:
                    if num <= __num_last:
                        alpha -= 1
                        self.valid = False
                        num += 1
                    else:
                        self.valid = True
                        break
            __err = 0.5
            for key in rad_dict.keys():
                valid, x0, y0, z0 = self.forward_kinematics(rad_dict[key][0], rad_dict[key][1], rad_dict[key][2],
                                                            rad_dict[key][3])
                if abs(x0 - x) > __err or abs(y0 - y) > __err or abs(z0 - z) > __err:
                    del rad_dict[key]
                    logger.warning('逆运动学带入正运动学公式误差大于{}'.format(__err))
                else:
                    pass
            self.rad_dict = rad_dict
            if is_return_one:
                rad = self.alpha_sort_out(self.rad_dict, max_alpha=max_alpha, min_alpha=min_alpha)
                self.rad = rad
                return rad
            else:
                return rad_dict

    def inverse_kinematics_limit_theta4(self, x, y, z, alpha_j4=45.66):
        x_orig = float(x)
        x = x_orig - self.magnet_dx

        y_orig = float(y)
        y = y_orig - self.magnet_dy

        z = float(z)

        self.valid = False
        # rad2 = None
        # rad3 = None
        # rad4 = None
        __MIN_ALPHA = 90  # j2+j3+j4 min value, 最后一个joint不向后仰
        # num = 0
        # __num_last = 50
        if z < 0:
            logger.warning('z 不能小于0')
            self.valid = False
            raise ValueError
        if y < 0:
            logger.warning('y 不能小于0')
            self.valid = False
            raise ValueError

        length = math.sqrt(pow((y + self.P), 2) + pow(x, 2))

        if length == 0:
            rad1 = 0  # 可以是任意数
        else:
            rad1 = math.atan2((y + self.P), x)

        height = z + self.A5

        if self.__valid_j(1, rad1) and not self.__out_of_range(length, height) and not self.valid:
            rad4 = math.radians(alpha_j4)
            l22_add_h22 = (self.A3 ** 2 + self.A4 ** 2 + 2 * self.A3 * self.A4 * math.cos(rad4))
            d1 = 2 * height * self.A2 - 2 * self.A1 * self.A2
            d2 = 2 * length * self.A2
            d0 = self.A2 ** 2 + self.A1 ** 2 + length ** 2 + height ** 2 - 2*self.A1*height - l22_add_h22

            # 定义方程
            def equation(_theta2, _d0, _d1, _d2):
                return _d1 * math.cos(_theta2) + _d2 * math.sin(_theta2) - _d0

            # 初始猜测值
            initial_guess = 0

            # 求解方程
            solution = newton(equation, initial_guess, args=(d0, d1, d2))
            logger.debug("j2的角度：{}".format(math.degrees(solution)))
            rad2 = solution
            if not self.__valid_j(2, rad2):
                logger.warning("j2角度{}不符合范围：".format(math.degrees(rad2)))
                return None
            l0 = length - self.A2*math.sin(rad2)
            h0 = height - self.A2*math.cos(rad2) - self.A1

            c1 = self.A3 + self.A4*math.cos(rad4)
            c2 = self.A4 * math.sin(rad4)

            rad23 = math.atan2((c1 * l0 - c2 * h0), (c1 * h0 + c2 * l0))
            rad3 = rad23 - rad2
            if not self.__valid_j(3, rad3):
                logger.warning("j3角度{}不符合范围：".format(math.degrees(rad2)))
                return None
            __err = 0.1
            rad6 = math.pi - rad2 - rad3 - rad4
            theta_list = [round(math.degrees(rad1), 2), round(math.degrees(rad2), 2),
                          round(math.degrees(rad3), 2), round(math.degrees(rad4), 2), 0, round(math.degrees(rad6), 2)]
            logger.debug("theta_list{}".format(theta_list))
            # x0 = x0 + self.magnet_dx

            valid, x0, y0, z0 = self.forward_kinematics(theta_list[0], theta_list[1], theta_list[2],
                                                        theta_list[3])
            logger.debug("x0:{}, y0:{}, z0:{}".format(x0, y0, z0))
            # z = z - self.A5
            err_x0 = abs(x0 - x_orig)
            err_y0 = abs(y0 - y_orig)
            err_z0 = abs(z0 - z)

            if abs(x0 - x_orig) <= __err and abs(y0 - y_orig) <= __err and 0 <= z0 - z <= __err:
                logger.info('succeed，逆运动学带入正运动学公式误差小于{}'.format(__err))
                logger.info("err_x0:{}, err_y0:{}, err_z0:{}".format(__err, err_x0, err_y0, err_z0))
                return theta_list
            else:
                logger.warning("false, 逆运动学带入正运动学公式误差大于{}, err_x0:{}, err_y0:{}, err_z0:{}".format(__err, err_x0, err_y0, err_z0))
                return None

        def get_alpha6(_theta_list):
            """
            返回末端执行器alpha6垂直的角度，输入6个theta角度，依次从底座开始，从下往上计算
            """
            _rad6 = math.pi - _theta_list[1] - _theta_list[2] - _theta_list[3]
            alpha6 = round(math.degrees(_rad6), 2)
            return alpha6


def theta_trans_pulse(_theta_list_or_dict, is_theta_dict=False):
    """
    :param _theta_list_or_dict:依次从底座，从下往上
    :param is_theta_dict:
    :return:
    """
    pulse5 = 500
    if is_theta_dict:
        _pulse_dict = {}
        for key in _theta_list_or_dict.keys():
            pulse1 = int(500 + ((_theta_list_or_dict[key][0] - 90) / 0.24))
            pulse2 = int(500 - (_theta_list_or_dict[key][1] / 0.24))
            pulse3 = int(500 + (_theta_list_or_dict[key][2] / 0.24))
            pulse4 = int(500 - (_theta_list_or_dict[key][3] / 0.24))
            pulse6 = int(500 + (_theta_list_or_dict[key][5] / 0.24))

            logger.info("pulse1:{}, pulse2:{}, pulse3:{}, pulse4:{},pulse5:{} pulse6{}".format(pulse1, pulse2,
                                                                                               pulse3, pulse4, pulse5,
                                                                                               pulse6))
            _pulse_dict[key] = [pulse1, pulse2, pulse3, pulse4, pulse5, pulse6]
        return _pulse_dict
    else:
        _pulse_list = []
        pulse1 = int(500 + ((_theta_list_or_dict[0] - 90) / 0.24))
        pulse2 = int(500 - (_theta_list_or_dict[1] / 0.24))
        pulse3 = int(500 + (_theta_list_or_dict[2] / 0.24))
        pulse4 = int(500 - (_theta_list_or_dict[3] / 0.24))
        pulse6 = int(500 + (_theta_list_or_dict[5] / 0.24))

        _pulse_list.append([pulse1, pulse2, pulse3, pulse4, pulse5, pulse6])
        logger.info("pulse1:{}, pulse2:{}, pulse3:{}, pulse4:{},pulse5:{} pulse6{}".format(pulse1, pulse2,
                                                                                           pulse3, pulse4, pulse5,
                                                                                           pulse6))
        return _pulse_list


def pulse_trans_theta(_pulse_list_or_dict, is_theta_dict=False):
    """
    :param is_theta_dict:依次从底座，从下往上
    :param _pulse_list_or_dict:对应
    :return:返回对应的角度
    """
    _theta5 = 0
    if not is_theta_dict:
        _theta6 = round(((_pulse_list_or_dict[5] - 500) * 0.24), 2)
        _theta4 = round(((500 - _pulse_list_or_dict[3]) * 0.24), 2)
        _theta3 = round(((_pulse_list_or_dict[2] - 500) * 0.24), 2)
        _theta2 = round(((500 - _pulse_list_or_dict[1]) * 0.24), 2)
        _theta1 = round(((_pulse_list_or_dict[0] - 500) * 0.24) + 90, 2)
        logger.info("theta1:{}, theta2:{}, theta3:{}, theta4:{},theta5:{}, theta6:{}".format(_theta1, _theta2, _theta3,
                                                                                             _theta4, _theta5, _theta6))
        return _theta1, _theta2, _theta3, _theta4, _theta5, _theta6
    else:
        _theta_dict = {}
        for key in _pulse_list_or_dict.keys():
            _theta6 = round(((_pulse_list_or_dict[5] - 500) * 0.24), 2)
            _theta4 = round(((500 - _pulse_list_or_dict[3]) * 0.24), 2)
            _theta3 = round(((_pulse_list_or_dict[2] - 500) * 0.24), 2)
            _theta2 = round(((500 - _pulse_list_or_dict[1]) * 0.24), 2)
            _theta1 = round(((_pulse_list_or_dict[0] - 500) * 0.24 + 90), 2)
            _theta_dict[key] = [_theta1, _theta2, _theta3, _theta4, _theta5, _theta6]
            logger.info("theta1:{}, theta2:{}, theta3:{}, theta4:{},theta5:{}, theta6:{}".format(_theta1, _theta2,
                                                                                                 _theta3, _theta4,
                                                                                                 _theta5, _theta6))
        return _theta_dict


def hiwonder_robot_trans_theta_id(pulse_list):
    plistmy = pulse_list[::-1]
    return plistmy


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # theta1, theta2, theta3, theta4, theta5, theta6 = pulse_trans_theta(
    #     hiwonder_robot_trans_theta_id([479, 499, 248, 508, 225, 578]))
    ik = RobotKinematics()
    # _, x1, y1, z1 = ik.forward_kinematics(theta1, theta2, theta3, theta4)
    # print("x1:{}, y1:{}, z1:{},".format(x1, y1, z1))
    #
    # # res = ik.inverse_kinematics_limit_theta4(x1, y1, z1, alpha_j4=theta4)
    # res = ik.inverse_kinematics_limit_theta4(0, 3, z1, alpha_j4=90)
    #
    # logger.debug(res)
    x1 = 2.21
    y1 = 31.83
    z = 2.2

    res = ik.inverse_kinematics_limit_theta4(x1, y1, z, alpha_j4=48.24)
    print("res{}".format(res))
    my_pwm_list = theta_trans_pulse(res)[0]
    print(my_pwm_list)
