import threading
import time

import cv2
import numpy as np

from chess_trans import recognition_circle_multiple
from chess_utils.image_processing import affirm_specifies_the_number_of_circles


class Test1:
    def __init__(self):

        self.frames_list_new = None
        self.frames_list = None
        self.is_allow_photos = False

    def run(self):
        photo_thread = threading.Thread(
            target=self.photo_thread,
            daemon=True,
            name="photo_thread"
        )
        self.is_allow_photos = True
        photo_thread.start()
        time.sleep(3)
        self.is_allow_photos = False
        # 等待子线程执行完毕
        photo_thread.join()
        print("执行完成")

    def photo_thread(self):

        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            print("无法打开摄像头")
            return None
        frames_list = []
        gray_image_list = []
        pixel_values_list = []
        current_pixel_values = 0
        current_gray_sum = 0
        times = 10
        numm = 0
        while self.is_allow_photos:
            ret, frame = cap.read()
            if not ret:
                print("未能从摄像头读取帧.")
                break
            # gray_image = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
            # 计算灰度总和
            # gray_sum = np.sum(gray_image)
            numm = numm + 1
            cv2.imwrite(
                "C:\\Users\\jhf12\\Documents\\graduation_project\\chess_robot\\output\\frames\\" + str(numm) + ".png",
                frame)
            frames_list.append(frame)
            # # # 统计像素值的分布
            # # pixel_values, counts = np.unique(gray_image, return_counts=True)
            # pixel_values_list.append(pixel_values)
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
                self.is_allow_photos = False
                print("请走棋")
                cap.release()
                break
            current_gray = gray_sum  # 更新 current_gray
        cap.release()


if __name__ == '__main__':
    test = Test1()
    test.run()
    print(len(test.frames_list))

    # res = affirm_specifies_the_number_of_circles(test.frames_list, 30, 3)
    # if res[0]:
    res_robot = recognition_circle_multiple(time=1, is_use_saved_picture=True, frame_list=test.frames_list)
    cv2.imshow("windows", test.frames_list[0])
    cv2.waitKey(0)
    # print(len(test.frames_list))
