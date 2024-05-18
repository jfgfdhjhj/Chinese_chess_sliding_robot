#!/usr/bin/env python3
# encoding:utf-8
import cv2
import numpy as np
from system import save_roi_parameter, roi_parameter_list

# 相机内参:
A_opencv = [[3.22420309e+03, 0.00000000e+00, 6.84955214e+02],
            [0.00000000e+00, 3.30962303e+03, 7.73265517e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
A_opencv = np.array(A_opencv, dtype=np.float16)
# 相机畸变(k1, k2, p1, p2, k3):
k_opencv = [0.22141523, -0.89428706, 0., 0., 0.]
k_opencv = np.array(k_opencv, dtype=np.float16)


def mouse_callback_official(event, x, y, flags, user_data):
    global cur_shape, start_pos
    image_user = user_data
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if cur_shape == 1:
            cv2.line(image_user, start_pos, (x, y), (0, 0, 255), 5, 5)
        elif cur_shape == 2:
            a0 = x - start_pos[0]
            b0 = y - start_pos[1]
            r0 = int((a0 ** 2 + b0 ** 2) ** 0.5)
            cv2.circle(image_user, start_pos, r0, (0, 0, 255), 5, 16)
        elif cur_shape == 3:
            cv2.rectangle(image_user, start_pos, (x, y), (0, 0, 255), 5, 16)
    else:
        print(event, x, y, flags)


FLAG_MOUSE_CLIK = 0
start_pos = (0, 0)
finish_pos = (0, 0)


def mouse_callback(event, x, y, flags, user_data):
    """
    鼠标回调函数，定义一次点击和二次点击，确定起始和开始坐标
    :param event: 鼠标回调函数的事件
    :param x: x坐标
    :param y: y坐标
    :param flags:
    :param user_data: 这里的数据为传入的图片
    :return: 无返回值
    """
    global FLAG_MOUSE_CLIK, start_pos, finish_pos
    click_first_down = 1
    click_first_up = 2
    click_second_down = 3
    image_user = user_data
    if event == cv2.EVENT_LBUTTONDOWN:
        if FLAG_MOUSE_CLIK == 0:
            FLAG_MOUSE_CLIK = click_first_down
            start_pos = (x, y)
        elif FLAG_MOUSE_CLIK == click_first_up:
            FLAG_MOUSE_CLIK = click_second_down
            finish_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if FLAG_MOUSE_CLIK == click_first_down:
            FLAG_MOUSE_CLIK = click_first_up
            print("起始坐标为：", start_pos)
        elif FLAG_MOUSE_CLIK == click_second_down:
            print("终止坐标为：", finish_pos)
            FLAG_MOUSE_CLIK = 0
    # print(event, x, y, flags, user_data)


def crop_picture(win_name: str, image_user):
    """
    裁剪出图片感兴趣的区域
    :param win_name: 窗口名称
    :param image_user: 要裁剪的图片
    :return: 裁剪完后的图片
    """
    global start_pos, finish_pos, picture_is_crop
    cv2.setMouseCallback(win_name, mouse_callback, image_user)
    cv2.imshow(win_name, image_user)
    cv2.waitKey(0)
    roi = image_user[start_pos[1]:finish_pos[1], start_pos[0]:finish_pos[0]]
    cv2.resizeWindow(win_name, finish_pos[0] - start_pos[0], finish_pos[1] - start_pos[1])
    cv2.imshow(win_name, roi)
    cv2.waitKey(0)
    picture_is_crop = True
    print(start_pos, finish_pos)
    save_roi_parameter(start_pos, finish_pos)

    return roi


def track_bar_callback(value):
    print(value)


def save_picture(is_open_camera=True):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if is_open_camera:
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window", 1920, 1080)
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()
        # 初始化计数器
        count = 25
        while True and is_open_camera:
            flag, image = cap.read()
            cv2.imshow('window', image)
            key = cv2.waitKey(0)
            # 如果按下 'a' 键，拍摄照片并保存
            if key == ord('a'):
                count += 1
                filename = f'output/photos_{count}.png'
                cv2.imwrite(filename, image)
                print(f'保存照片 {filename}')
            # 如果按下 'q' 键，退出循环
            if key == ord('q'):
                break
    else:
        print("用户设置不打开摄像头")


def distortion_correction(img):
    """
    :return:
    """
    # 04 移除图像畸变
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    # 畸变校正，提前计算映射矩阵
    map_opencv_x, map_opencv_y = cv2.initUndistortRectifyMap(
        A_opencv, k_opencv, np.eye(3), A_opencv, (w, h), 5)
    # 校正后的图片相当于没有畸变了，直接用相机内参即可，无需k1, k2
    img_opencv = cv2.remap(img, map_opencv_x, map_opencv_y, cv2.INTER_LINEAR)
    return img_opencv


def taking_photos(is_open_camera=False):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if is_open_camera:
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window", 1920, 1080)
        if not cap.isOpened():
            print("无法打开摄像头")
            exit()
        while True and is_open_camera:
            flag, image = cap.read()
            cv2.imshow('window', image)
            key = cv2.waitKey()
            # 如果按下 'a' 键，拍摄照片并保存
            if key == ord('a'):
                return image
            # 如果按下 'q' 键，退出循环
            if key == ord('q'):
                break

    else:
        print("用户设置不打开摄像头")


def one_frame_grab(is_crop=False):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None
    else:
        flag, image = cap.read()
        if is_crop:
            # image = distortion_correction(image)
            image = image[roi_parameter_list[0][1]:roi_parameter_list[1][1], roi_parameter_list[0][0]:roi_parameter_list[1][0]]
        cap.release()
        return image


def frame_grab(num_frames=5, is_crop=False):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None
    else:
        frames_list = []
        frame_count = 0
        while frame_count < num_frames:
            ret, frame = cap.read()
            if is_crop:
                # frame = distortion_correction(frame)
                frame = frame[roi_parameter_list[0][1]:roi_parameter_list[1][1], roi_parameter_list[0][0]:roi_parameter_list[1][0]]
            if not ret:
                print("未能从摄像头读取帧.")
                break
            frames_list.append(frame)
            frame_count += 1
        cap.release()
        return frames_list


if __name__ == "__main__":
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    image = one_frame_grab(is_crop=False)
    crop_picture("window", image)

    # img = one_frame_grab(is_crop=True)
    img = frame_grab(is_crop=True)[0]
    cv2.imshow("windows", img)
    cv2.waitKey()

    # save_picture()
    # cap = cv2.VideoCapture(1)
    # cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("window", 1920, 1080)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # while True:
    #     flag, image = cap.read()
    #     key = cv2.waitKey(0)
    #     # cv2.imshow('window', image)
    #     if key == ord('a'):
    #         opencv_image = distortion_correction(image)
    #         cv2.imshow('window', opencv_image)
    #         print("图像校准成功")
    #     # 如果按下 'q' 键，退出循环
    #     if key == ord('q'):
    #         break

    # image = one_frame_grab()
    # opencv_image = distortion_correction(image)
    # cv2.imshow('window', opencv_image)
    # cv2.waitKey()
    # print("图像校准成功")

    # frames = frame_grab(num_frames=5)
    # cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("window", 1920, 1080)
    # for i, fruit in enumerate(frames, start=1):
    #     cv2.imshow(f'Image{i}', fruit)
    # cv2.waitKey(0)
