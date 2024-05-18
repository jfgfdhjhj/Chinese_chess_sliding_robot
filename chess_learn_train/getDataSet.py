import numpy as np
import cv2
from chess_utils.image_processing import ImageToFont
import os

from Utils import *
# 这里修改序号，范围从0到6
list_num = "6"
# 黑色为True， 红色为False
is_black = True
# 0--黑将, 1--黑車, 2--黑馬, 3--黑象, 4--黑士,  5--黑炮,  6--黑卒,
# 0--红帅,  1--红車, 2--红马, 3--红相, 4--红士, 5--红炮, 6--红兵,

# 第几组数据,从1开始,一组数据有250个
num_label = 4
# 数据从几号开始编号
sample_num = (num_label - 1) * 250
# sample_num = 960
sample_num_lat = num_label * 250
# 摄像头测试
DEBUG = 0

FLAG_MOUSE_CLIK = 0
start_pos = (0, 0)
finish_pos = (0, 0)
picture_is_crop = True
userChessNumCount = 0
pcChessNumCount = 0

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
print(current_dir)
datasets_path = os.path.join(current_dir, "datasets")
photos_path = "/photos"

data_path = "/chess_learn_train/datasets"

train_path = datasets_path + "\\train"
test_path = datasets_path + "\\test"

if is_black:
    test_path_num = test_path + "\\black\\{}\\".format(list_num)
    train_path_red_black = train_path + "\\black\\{}\\".format(list_num)
else:
    test_path_num = test_path + "\\red\\{}\\".format(list_num)
    train_path_red_black = train_path + "\\red\\{}\\".format(list_num)

cap = cv2.VideoCapture(1)
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 1920, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_FPS, 30)


# 霍夫圆检测阈值滑动条回调函数
def hough_circles_callback(value):
    print("value")


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

    return roi


while cap.isOpened() & DEBUG:
    flag, frame = cap.read()
    if not flag:
        print("failed")
        break
    else:
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
        elif key & 0xff == ord("a"):
            cv2.imwrite(photos_path + "test1" + ".jpg", frame)
            print("打印成功")
        else:
            cv2.imshow("window", frame)
# image = cv2.imread(photos_path + "test1" + ".jpg")

# 创建霍夫圆检测阈值滑动条
cv2.createTrackbar('HoughCircles', 'window', 100, 1500, hough_circles_callback)

# 创建一个 44x44 的核
imgChess44_44Kernel = np.array([[False if i+j < 13 or i+j > 81 or i > j+23 or j > i+23 else True for j in range(44)]
                                for i in range(44)])


def main():
    global sample_num, sample_num_lat
    while cap.isOpened():
        # global test_path_num, train_black_num
        flag, image = cap.read()
        if not picture_is_crop:
            image = crop_picture("window", image)
            break
        if not flag:
            print("failed")
            break
        else:
            a = cv2.getTrackbarPos('HoughCircles', "window")
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 中值滤波
            img_median = cv2.medianBlur(gray, 3)
            # 高斯滤波
            # fil2 = cv2.GaussianBlur(image, (3, 3), sigmaX=1)
            # 直方图均衡化
            imgEqualizeHist = cv2.equalizeHist(img_median)
            # 霍夫圆检测
            circles = cv2.HoughCircles(
                imgEqualizeHist,
                cv2.HOUGH_GRADIENT, dp=1, minDist=32, param1=a, param2=19, minRadius=28, maxRadius=32
            )
            key = cv2.waitKey(500)
            if key & 0xff == ord('p'):
                break
            # key & 0xff == ord("a"):
            elif sample_num >= sample_num_lat:
                print("采集完成")
                break
            else:
                # 如果检测到圆
                if circles is not None:
                    sample_num += 1
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        x = i[0]
                        y = i[1]
                        r1 = i[2]
                        print(r1)
                        # r1 = 19

                        # 画出圆的外接圆
                        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

                        # 画出圆心
                        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

                        # 裁剪正方形
                        r = r1
                        cut_r = gray[i[1] - r:i[1] + r, i[0] - r:i[0] + r]
                        image_to_font = ImageToFont(cut_r)
                        image_font = image_to_font()

                        if sample_num <= sample_num_lat - 20:
                            cv2.imwrite(train_path_red_black + str(sample_num) + ".png", image_font)
                            print(train_path_red_black + str(sample_num) + ".png")
                            print("训练集{0}存入成功".format(sample_num))
                        else:
                            cv2.imwrite(test_path_num + str(sample_num) + ".png", image_font)
                            print("测试集{0}存入成功".format(sample_num))

                        cv2.imshow("window", image)
                else:
                    print("没有检测到圆")
                    continue

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
