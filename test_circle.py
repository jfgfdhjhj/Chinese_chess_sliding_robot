import json

import cv2
from camera import one_frame_grab
from chess_utils.image_processing import statistical_red_pixel
from system import roi_parameter_list, hough_circles_parameter_path, hough_circles_parameter_dict


def hough_circles_callback(value):
    hough_circles_parameter_dict['param1'] = value
    print("霍夫圆阈值检测参数{}".format(value))


# 红黑棋子分类阈值滑动条回调函数
def color_classify_callback(value):
    hough_circles_parameter_dict["color"] = value
    print("红黑棋子分类阈值检测参数{}".format(value))


def hough_circles_param2_callback(value):
    hough_circles_parameter_dict["param2"] = value
    print("中心点累加器阈值参数{}".format(value))


def hough_circles_r_min_callback(value):
    hough_circles_parameter_dict['r_min'] = value
    print("圆最小半径阈值参数{}".format(value))


def hough_circles_r_max_callback(value):
    hough_circles_parameter_dict['r_max'] = value
    print("圆最大半径阈值参数{}".format(value))


def minDist_callback(value):
    print("minDist_callback{}".format(value))


cv2.namedWindow("slider", cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow("slider", roi_parameter_list[1][0] - roi_parameter_list[0][0], roi_parameter_list[1][1] - roi_parameter_list[0][1])


# 创建霍夫圆检测阈值滑动条
cv2.createTrackbar('r_min', 'slider', hough_circles_parameter_dict["r_min"], 50, hough_circles_r_min_callback)

# 创建中心点累加器阈值滑动条
cv2.createTrackbar('r_max', 'slider', hough_circles_parameter_dict["r_max"], 50, hough_circles_r_max_callback)

# 创建霍夫圆检测阈值滑动条
cv2.createTrackbar('HoughCircles', 'slider', hough_circles_parameter_dict["param1"], 255, hough_circles_callback)

# 创建中心点累加器阈值滑动条
cv2.createTrackbar('HoughCircles_param2', 'slider', hough_circles_parameter_dict["param2"], 255,
                   hough_circles_param2_callback)

cv2.createTrackbar('minDist', 'slider', hough_circles_parameter_dict["param2"], 255,
                   minDist_callback)


cv2.createTrackbar('color_classify', 'slider', hough_circles_parameter_dict["color"], 1000,
                   color_classify_callback)

image = one_frame_grab()
# cv2.imshow('image', image)
# cv2.waitKey(0)

while True and 1:

    # cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # flag, image = cap.read()
    image = one_frame_grab()

    imgSource = image.copy()

    r_min = cv2.getTrackbarPos('r_min', "slider")
    r_max = cv2.getTrackbarPos('r_max', "slider")
    a = cv2.getTrackbarPos('HoughCircles', "slider")
    b = cv2.getTrackbarPos('color_classify', "slider")
    c = cv2.getTrackbarPos('HoughCircles_param2', "slider")
    d = cv2.getTrackbarPos('minDist', "slider")
    # 转为灰度图
    gray = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)

    # 中值滤波
    imgMedian = cv2.medianBlur(gray, 3)

    # 直方图均衡化
    imgEqualizeHist = cv2.equalizeHist(imgMedian)

    # # 二值化
    # thresh, threshold_two = cv2.threshold(imgEqualizeHist, 100, 255, cv2.THRESH_BINARY)

    # 输入图像，方法（类型），dp(dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推)，# 38
    # 最短距离-可以分辨是两个圆否 则认为是同心圆 ,边缘检测时使用Canny算子的高阈值，中心点累加器阈值—候选圆心（霍夫空间内累加和大于该阈值的点就对应于圆心），检测到圆的最小半径，检测到圆的的最大半径
    # 用a代替param1, 用c代替param2 # 27
    circles = cv2.HoughCircles(
        imgEqualizeHist, cv2.HOUGH_GRADIENT, dp=1, minDist=38, param1=a, param2=c, minRadius=r_min, maxRadius=r_max
    )

    # 检测到了圆
    if circles is not None:
        # 遍历每一个圆
        for circle in circles[0]:
            x, y, r1 = int(circle[0]), int(circle[1]), int(circle[2])
            redPixelValueCount = statistical_red_pixel(imgSource[y - r1:y + r1, x - r1:x + r1])
            if redPixelValueCount > b:
                cv2.circle(imgSource, (x, y), r1, (0, 255, 0), 2)
            else:
                cv2.circle(imgSource, (x, y), r1, (0, 0, 255), 2)
    cv2.imshow("slider", imgSource)
    key = cv2.waitKey(100)
    if key == ord("q"):
        print(hough_circles_parameter_dict)
        with open(hough_circles_parameter_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(hough_circles_parameter_dict))
            print("霍尔圆参数保存完成!")
        break
