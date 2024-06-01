import re
import time
import traceback
import serial
import threading
import queue

from chess_trans import fen_move_trans_key_id
from chess_utils.logger import logger
from system import slideway_coordinates_dict
from slideway.mapping import HandInEyeCalibrationSlide, dar_x, dar_y, offset_y
import serial.tools.list_ports


class SerialParser:
    def __init__(self, port='COM5', baudrate=115200, timeout=1):
        self.running = None

        plist = list(serial.tools.list_ports.comports())
        if len(plist) <= 0:
            print("The Serial port can't find!")
            return
        else:
            plist_0 = list(plist[0])
            serialName = plist_0[0]

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        if self.ser.is_open:
            print("串口已打开")
            time.sleep(2)

        else:
            print("串口打开失败")
            return

        # 等待一段时间确保连接建立
        self.queue = queue.Queue()
        self.parser_thread = threading.Thread(
            target=self.parser,
            daemon=True,
            name="ParserThread"
        )
        self.parser_thread.start()
        self.thread = threading.Thread(target=self.serial_reader, daemon=True)
        self.thread.start()

    def serial_reader(self):
        while True:
            if self.ser.in_waiting:
                data = self.ser.readline().decode('utf-8').strip()
                self.queue.put(data)

    def parser(self):
        # 解析线程
        self.running = True
        while self.running:
            line = self.queue.get()
            try:
                self.parse_line(line)
            except Exception:
                logger.error(traceback.format_exc())

    def parse_line(self, line):
        """
        解析，由子类实现
        """
        print(line)

    def serial_close(self):
        self.ser.close()

    def send_command(self, _command):
        self.ser.write(bytes(_command + "\n", encoding='utf-8'))
        logger.info("发送命令：{}".format(_command))


class SlideSerialParser(SerialParser):
    # x_a = 3000
    # y_a = 3000

    # x_rate = 4000
    # y_rate = 4000
    # z_rate = 1000

    def __init__(self, port='COM6', baudrate=115200, timeout=1):
        super().__init__(port, baudrate, timeout)

        self.rate_dict = {}
        # self.x_rate = None
        # self.y_rate = None
        # self.z_rate = None
        self.a_dict = {}
        # self.x_a = None
        # self.y_a = None
        # self.z_a = None

        self.y_pos_idle = None
        self.z_pos_idle = None
        self.x_pos_idle = None

        self.z_pos_run = None
        self.y_pos_run = None
        self.x_pos_run = None

        self.is_ok = False
        self.slide_is_running = False

        self.position_dict = {}
        # self.x_position = None
        # self.y_position = None
        self.z_position = None

        self.find_rate()

    def find_rate(self):
        self.send_command("$110")
        self.send_command("$111")
        self.send_command("$112")

        self.send_command("$120")
        self.send_command("$121")
        self.send_command("$122")

    def check_finish(self, time_out=15):
        time_num = 0
        try:
            while True:
                self.send_command("?")
                time.sleep(0.1)
                while self.is_ok:
                    self.is_ok = False
                    time.sleep(1)
                    time_num += 1
                    self.send_command("?")
                    time.sleep(0.1)
                    if time_num >= time_out:
                        logger.warning("在规定时间内，没有到达指定位置，失败")
                        return False
                    elif self.slide_is_running:
                        logger.info("机器运行中")
                        continue
                    else:
                        logger.info("机器停止")
                        return True
                break
        except:
            pass

    def calculating_motion_time(self, d, a, v):
        motion_time = v/a + d/v

        return motion_time

    def check_position(self, *target_position, time_out=15, tolerance=0.1, is_check_ok=True):
        """
        检查位置，*target_position为元组，例如（"x":5），（"y":10)，（"z":20）
        """
        # 初始化目标位置
        target_positions = {}
        target_positions_time = {}
        time_sleep_list = []
        # 解析每个元组，将坐标轴和目标位置存储到字典中
        for pos in target_position:
            if len(pos) != 2:
                logger.error("传入的参数格式不正确")
                return False
            axis, coord = pos
            target_positions[axis] = coord

        time_num = 0
        errors = {}

        for pos in target_position:
            axis, coord = pos
            data_d = abs(self.position_dict[axis] - target_positions[axis])
            time_sleep = target_positions_time[axis] = self.calculating_motion_time(data_d, self.a_dict[axis], self.rate_dict[axis])
            time_sleep_list.append(time_sleep)
        logger.info("滑轨最长移动时间{}".format(max(time_sleep_list)))
        time.sleep(max(time_sleep_list))

        for pos in target_position:
            axis, coord = pos
            self.position_dict[axis] = coord
        if not is_check_ok:
            return True
        # if self.position_dict["x"] == 0 and self.position_dict["y"] == 0:
        #     logger.info("理论可以走棋点")
        else:
            try:
                while True:
                    self.send_command("?")
                    time.sleep(0.1)
                    while self.is_ok:
                        self.is_ok = False
                        time_num += 1
                        self.send_command("?")
                        time.sleep(0.1)
                        if time_num >= time_out:
                            # if all(error <= 0.1 for error in errors.values()):
                            #     logger.debug("到达指定位置, 误差小于等于0.1")
                            logger.warning("在规定时间内，没有到达指定位置，失败")
                            for axis, target_pos in target_positions.items():
                                current_pos = getattr(self, f"{axis}_pos_run")
                                if current_pos is not None:
                                    errors[axis] = abs(target_pos - current_pos)
                            logger.warning("误差{}".format(errors.values()))
                            return False
                        elif self.slide_is_running:
                            logger.info("机器运行中")
                            continue
                        else:
                            logger.info("机器停止")
                            return True
                    break
            except:
                pass
            # # 计算误差
            # errors = {}
            # for axis, target_pos in target_positions.items():
            #     current_pos = getattr(self, f"{axis}_pos_run")
            #     if current_pos is not None:
            #         errors[axis] = abs(target_pos - current_pos)
            # logger.info("误差{}".format(errors.values()))
            # if all(error <= 0.1 for error in errors.values()):
            #     logger.debug("到达指定位置, 误差小于等于0.1")
            #     return True
            # elif time_num >= time_out:
            #     logger.warning("在规定时间内，没有到达指定位置，失败")
            #     return False
            # else:
            #     break

    def run_assigned_position(self, *target_position, time_out=15, is_check_ok=True):
        """
        运动到指定位置，定义*target_position为元组，格式为（"x",2),("y",-2),("z",-5)
        其中X向左为正方向，Y上为正方向，Z向下运动为正方向
        """

        # 初始化命令字符串
        move_command = "G0 "
        self.is_ok = False
        # 解析每个元组，将坐标轴和目标位置存储到字典中
        for pos in target_position:
            if len(pos) != 2:
                logger.error("传入的参数格式不正确")
                return False
            axis, coord = pos
            move_command += f"{axis.upper()}{coord} "
        self.send_command(move_command)
        res = self.check_position(*target_position, time_out=time_out, is_check_ok=is_check_ok)

        if res:
            logger.info("移动到指定点")
            return True
        else:
            logger.warning("没有移动到指定点")
            return False

    def parse_line(self, line):
        ok_pattern = re.compile(r'ok')
        # 使用正则表达式搜索字符串
        match_ok = ok_pattern.search(line)
        num_pattern = r'\$(\d+)(?:=(\d+\.\d{3}))?'
        num_matches = re.match(num_pattern, line)
        # 使用正则表达式分别匹配"Idle"和"Run"
        idle_match = re.search(r"<Idle\|MPos:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)\|", line)
        run_match = re.search(r"<Run\|MPos:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)\|", line)

        if idle_match:
            self.slide_is_running = False
            x_pos_idle = float(idle_match.group(1))
            y_pos_idle = float(idle_match.group(2))
            z_pos_idle = float(idle_match.group(3))
            logger.info("Idle状态下的坐标：")
            logger.info("X轴坐标:{}, Y轴坐标:{}, Z轴坐标:{}".format(x_pos_idle, y_pos_idle, z_pos_idle))

            self.x_pos_idle = x_pos_idle
            self.y_pos_idle = y_pos_idle
            self.z_pos_idle = z_pos_idle

            self.x_position = self.x_pos_idle
            self.y_position = self.y_pos_idle
            # self.z_position = self.z_pos_idle

        elif run_match:
            self.slide_is_running = True
            x_pos_run = float(run_match.group(1))
            y_pos_run = float(run_match.group(2))
            z_pos_run = float(run_match.group(3))
            logger.info("\nRun状态下的坐标：")
            logger.info("X轴坐标:{}, Y轴坐标:{}, Z轴坐标:{}".format(x_pos_run, y_pos_run, z_pos_run))

            self.x_pos_run = x_pos_run
            self.y_pos_run = y_pos_run
            self.z_pos_run = z_pos_run

        elif match_ok:
            logger.info("匹配成功:{}".format(match_ok.group()))
            logger.debug("ok")
            self.is_ok = True

        elif num_matches:
            dollar_value = num_matches.group(1)
            decimal_value = num_matches.group(2)

            if dollar_value == "110":
                self.rate_dict["x"] = float(decimal_value)/60
                logger.info("当前x_rate值为:{}".format(decimal_value))
            elif dollar_value == "111":
                self.rate_dict["y"] = float(decimal_value)/60
                logger.info("当前y_rate值为:{}".format(decimal_value))
            elif dollar_value == "112":
                self.rate_dict["z"] = float(decimal_value)/60
                logger.info("当前z_rate值为:{}".format(decimal_value))

            elif dollar_value == "120":
                self.a_dict["x"] = float(decimal_value)
                logger.info("当前x_a值为:{}".format(decimal_value))
                print()
            elif dollar_value == "121":
                self.a_dict["y"] = float(decimal_value)
                logger.info("当前y_a值为:{}".format(decimal_value))
            elif dollar_value == "122":
                self.a_dict["z"] = float(decimal_value)
                logger.info("当前z_a值为:{}".format(decimal_value))

            # print("提取的 $ 值:", dollar_value)
            # print("提取的小数值:", decimal_value)

        else:
            logger.warning("没有找到匹配的部分")
            print(line)

    def set_home(self):
        self.is_ok = False
        self.send_command("G92 X0 Y0 Z0")
        self.position_dict["x"] = 0
        self.position_dict["y"] = 0
        self.position_dict["z"] = 0

    def return_to_home(self):
        self.run_assigned_position(("x", 0), ("y", 0), time_out=15)
        self.run_assigned_position(("z", 0), time_out=5, is_check_ok=False)
        self.z_position = 0

    def init(self):
        self.run_assigned_position(("x", 0), ("y", 0), ("z", 0), time_out=15)
        self.z_position = 0

    def uplift(self, z=0, speed=400):
        self.is_ok = False
        if z != 0:
            z = -abs(z)
        dz = abs(self.z_position - z)
        # 初始化命令字符串
        move_command = "G1 " + f"Z{z}" + f" F{speed}"
        self.send_command(move_command)
        speed = speed / 60
        time.sleep(dz / speed)
        logger.debug("机器z轴抬升时间：{}".format(dz / speed))
        self.z_position = z
        # self.check_finish(time_out=1)

    def fall_down(self, z=5, speed=400):
        self.is_ok = False
        if z != 0:
            z = abs(z)
        dz = abs(self.z_position - z)
        # 初始化命令字符串
        move_command = "G1 " + f"Z{z}" + f" F{speed}"
        self.send_command(move_command)
        speed = speed / 60
        time.sleep(dz / speed)
        logger.debug("机器z轴降落时间：{}".format(dz / speed))

        self.z_position = z
        # self.check_finish(time_out=1)


class SlideChessRobot(SlideSerialParser):
    # 正对滑轨方向单个棋盘格子的高
    high_y = space_between_y = dar_x
    # 正对滑轨方向单个棋盘格子的宽
    weight_x = space_between_x = dar_y

    # 棋子的厚度
    delta_z = -10
    delta_x = weight_x
    delta_y = high_y

    offset = offset_y

    # 单个棋子半径，单位mm
    chess_r = 7.5
    # 初始化棋盘数组,建立一个数组，x轴长8个棋子，y轴长2个棋子，z轴高2个棋子

    # 定义每个轴上的间隔数
    num_points_x = 8
    num_points_y = 2
    num_points_z = 2

    chess_eat_position = []

    for z in range(num_points_z):
        for y in range(num_points_y):
            for x in range(num_points_x):
                point_index = z * num_points_y * num_points_x + y * num_points_x + x
                coordinate = [(x + 1) * delta_x, y * delta_y, z * delta_z]
                chess_eat_position.append(coordinate)

    def __init__(self, port='COM7', baudrate=115200, timeout=2):
        super().__init__(port, baudrate, timeout)
        self.hand_eye_slide = HandInEyeCalibrationSlide()
        self.recovery_list = []
        self.eat_chess_num = -1
        self.is_eat_num_full = False

    def judge_add_one_eat_num(self):
        self.eat_chess_num += 1
        if self.eat_chess_num >= 32:
            self.is_eat_num_full = True

    def move_chess_id(self, move_id):
        """
        机器移动到指定的id点，这个点是定死的，存储在字典里
        """
        x_target = slideway_coordinates_dict[move_id][2]
        y_target = slideway_coordinates_dict[move_id][3]
        res = self.run_assigned_position(("x", x_target), ("y", y_target))
        if res:
            logger.info("移动成功,移动的点为{}".format(move_id))
            return True
        else:
            logger.warning("移动失败")
            return False

    def move_chess_pixel_x_y(self, pixel_x, pixel_y):
        """
        机器移动到指定像素点，这个点是通过照片映射出来的
        """
        res = self.hand_eye_slide.get_points_slide_limit_x_y(pixel_x, pixel_y)
        if not res[0]:
            logger.error("超出滑轨移动范围！")
            return False
        x_target = res[1]
        y_target = res[2]
        res = self.run_assigned_position(("x", x_target), ("y", y_target))
        if res:
            logger.info("移动成功,移动的点为x:{},y:{}".format(x_target, y_target))
            return True
        else:
            logger.warning("移动失败")
            return False

    def move_to_the_stack_site(self):
        target_x = self.chess_eat_position[self.eat_chess_num][0]
        target_y = self.chess_eat_position[self.eat_chess_num][1]
        target_z = self.chess_eat_position[self.eat_chess_num][2]
        z = abs(target_z) + abs(self.delta_z)
        self.uplift(z)
        self.run_assigned_position(("x", target_x), ("y", target_y), time_out=15)

    def move_fen_move(self, fen_move, is_eat=False):
        before_key_id, after_key_id = fen_move_trans_key_id(fen_move)
        if is_eat:
            self.init()
            self.judge_add_one_eat_num()
            if self.is_eat_num_full:
                return False
            self.move_chess_id(after_key_id)
            self.fall_down()
            time.sleep(2)
            self.move_to_the_stack_site()

            self.move_chess_id(before_key_id)
            self.fall_down()
            time.sleep(2)
            self.uplift(z=self.delta_z)

            self.move_chess_id(after_key_id)
            self.uplift()
            time.sleep(2)
            self.return_to_home()

        else:
            self.init()
            self.move_chess_id(before_key_id)
            self.fall_down()
            time.sleep(2)
            self.uplift(z=self.delta_z)

            self.move_chess_id(after_key_id)
            time.sleep(2)
            self.return_to_home()


if __name__ == '__main__':

    # Example usage:
    # chess_slide = SlideSerialParser()
    chess_slide = SlideChessRobot(port="COM6")
    # chess_slide = SerialParser()
    try:
        while True:
            key = input("请输入指令：")
            # chess_slide.return_to_home()
            chess_slide.move_fen_move(key)
            # chess_slide.run_assigned_position(("x"))
            # chess_slide.send_command("?")
    except KeyboardInterrupt:
        chess_slide.serial_close()
