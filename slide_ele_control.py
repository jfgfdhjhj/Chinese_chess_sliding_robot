import threading
import time
import cv2
import numpy as np

from chess_trans import fen_move_trans_key_id
from slideway.slideway_serial import SlideChessRobot
from socket_client_ele import ClientEle


class ChessRobotEle(SlideChessRobot):
    def __init__(self):
        super().__init__()
        self.frames_list_new = None
        self.frames_list = None
        self.is_allow_photos = False
        self.clint_ele = ClientEle(_is_vm=False)
        
    def move_fen_move(self, fen_move, is_eat=False):
        before_key_id, after_key_id = fen_move_trans_key_id(fen_move)
        if is_eat:
            self.init()

            self.judge_add_one_eat_num()
            if self.is_eat_num_full:
                return False

            self.move_chess_id(after_key_id)
            self.fall_down(z=2)
            self.clint_ele.send_electromagnet_attract()
            time.sleep(0.2)
            # time.sleep(2)
            self.move_to_the_stack_site()
            self.clint_ele.send_electromagnet_fall()
            time.sleep(0.5)

            self.move_chess_id(before_key_id)
            self.clint_ele.send_electromagnet_attract()
            time.sleep(0.2)
            self.fall_down(z=2)
            # time.sleep(2)
            self.uplift(z=self.delta_z)

            self.move_chess_id(after_key_id)
            self.uplift()
            time.sleep(0.2)
            self.clint_ele.send_electromagnet_fall()

            self.return_to_home()

        else:
            self.init()
            self.move_chess_id(before_key_id)
            self.clint_ele.send_electromagnet_attract()
            time.sleep(0.2)
            self.fall_down(z=2)
            self.uplift(z=self.delta_z)

            self.move_chess_id(after_key_id)
            self.uplift()
            time.sleep(0.2)
            self.clint_ele.send_electromagnet_fall()
            self.return_to_home()

    def move_pixel_to_another_fixed_point(self, before_pixel, after_pixel, fen_move, is_eat=False):
        photo_thread = threading.Thread(
            target=self.photo_thread,
            daemon=True,
            name="photo_thread"
        )
        before_key_id, after_key_id = fen_move_trans_key_id(fen_move)
        if is_eat:
            self.init()

            self.judge_add_one_eat_num()
            if self.is_eat_num_full:
                return False

            self.move_chess_pixel_x_y(after_pixel[0], after_pixel[1])
            self.fall_down(z=4)
            self.clint_ele.send_electromagnet_attract()
            time.sleep(0.2)
            self.move_to_the_stack_site()

            target_z = self.chess_eat_position[self.eat_chess_num][2]
            self.fall_down(abs(target_z))
            time.sleep(0.5)
            self.clint_ele.send_electromagnet_fall()
            time.sleep(0.2)
            self.uplift(abs(target_z))

            self.move_chess_pixel_x_y(before_pixel[0], before_pixel[1])
            self.fall_down(z=4)
            self.clint_ele.send_electromagnet_attract()
            time.sleep(0.2)
            self.uplift(z=self.delta_z)

            self.move_chess_id(after_key_id)
            self.uplift()
            time.sleep(0.5)
            self.clint_ele.send_electromagnet_fall()
            time.sleep(0.2)

            self.is_allow_photos = True
            photo_thread.start()
            self.return_to_home()

            self.is_allow_photos = False
            # 等待子线程执行完毕
            photo_thread.join()

        else:
            self.init()
            self.move_chess_pixel_x_y(before_pixel[0], before_pixel[1])
            self.fall_down(z=4)
            self.clint_ele.send_electromagnet_attract()
            time.sleep(0.2)
            self.uplift(z=self.delta_z)

            self.move_chess_id(after_key_id)
            self.uplift()
            time.sleep(0.5)
            self.clint_ele.send_electromagnet_fall()
            time.sleep(0.2)

            self.is_allow_photos = True
            photo_thread.start()
            self.return_to_home()

            # 等待子线程执行完毕
            self.is_allow_photos = False
            photo_thread.join()

    def photo_thread(self, delay=0):
        numm = 0
        time.sleep(delay)
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            print("无法打开摄像头")
            return None
        frames_list = []
        while self.is_allow_photos:
            ret, frame = cap.read()
            if not ret:
                print("未能从摄像头读取帧.")
                break
            numm = numm + 1
            frames_list.append(frame)
            cv2.imwrite("C:\\Users\\jhf12\\Documents\\graduation_project\\chess_robot\\output\\frames\\" + str(numm) + ".png", frame)
        cap.release()
        self.frames_list = frames_list[-3:]

    def photo_thread_multiframe_detection(self, delay=0):
        time.sleep(delay)
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        current_gray = 0  # 将初始化移到循环外部
        time_num = 0

        if not cap.isOpened():
            print("无法打开摄像头")
            return None
        frames_list = []
        while self.is_allow_photos:
            ret, frame = cap.read()
            if not ret:
                print("未能从摄像头读取帧.")
                break
            # 转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 计算灰度总和
            gray_sum = int(np.sum(gray))
            print("a：{}".format(gray_sum))
            print("b:{}".format(current_gray))

            if abs(current_gray - gray_sum) <= 900000:
                print("c:{}".format(abs(current_gray - gray_sum)))
                frames_list.append(frame)
                time_num += 1
            else:
                frames_list = []
                time_num = 0
            if time_num >= 10:
                self.frames_list_new = frames_list
                print("请走棋")
                cap.release()
                break
            current_gray = gray_sum  # 更新 current_gray
        cap.release()


class EndGameRoute(ChessRobotEle, ):
    temporarily_set_aside_list = []

    def __init__(self, fen=None):
        super().__init__()
        self.temporarily_set_aside_num = 0
        self.generate_temporarily_set_aside_list()

    def generate_temporarily_set_aside_list(self):
        for z in range(self.num_points_z):
            for y in range(self.num_points_y):
                for x in range(self.num_points_x):
                    point_index = z * self.num_points_y * self.num_points_x + y * self.num_points_x + x
                    coordinate = [(x + 1) * self.delta_x, y * self.delta_y, z * self.delta_z]
                    self.temporarily_set_aside_list.append(coordinate)


if __name__ == '__main__':
    chess_robot = ChessRobotEle()
    input("请设置原点")
    chess_robot.set_home()
    try:
        while True:
            key = input("请输入指令：")
            # chess_slide.return_to_home()
            chess_robot.move_fen_move(key, is_eat=False)
            # chess_slide.run_assigned_position(("x"))
            # chess_slide.send_command("?")
    except KeyboardInterrupt:
        chess_robot.serial_close()
