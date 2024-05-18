import json
import math
import cv2
import numpy as np
from tqdm import tqdm
from chess_utils.logger import logger
from system import hough_circles_parameter_dict, set_x_y_pixel_limit

_mask = np.empty((62, 62), dtype=bool)
_mask.fill(False)

for i in range(62):
    for j in range(62):
        if np.hypot((i - 31), (j - 31)) > 31:
            _mask[i, j] = True

file_board = r"C:\Users\jhf12\Documents\graduation_project\chess_robot\board_position.txt"
file_chess = r"C:\Users\jhf12\Documents\graduation_project\chess_robot\Checkerboard_positioning.txt"

try:
    with open(file_chess, 'r', encoding='utf-8') as f:
        chess_dic = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    logger.error("转换后的象棋位置文件不存在")

try:
    with open(file_board, 'r', encoding='utf-8') as f:
        chess_board_dic = json.loads(f.read())
        chess_board_position_list = chess_board_dic["chess_board"]
except FileNotFoundError:  # 抛出文件不存在异常
    logger.error("转换后的棋盘位置文件不存在")


class ImageToFont:
    def __init__(self, img_source):
        self.imgSource = img_source
        self.isCheck = True

        center = [0, 0]
        r3 = 0

        mask = np.zeros_like(self.imgSource)
        # mask[:] = 0
        # 反转掩码
        mask_inverted = cv2.bitwise_not(mask)
        # 霍夫圆检测
        circles1 = cv2.HoughCircles(self.imgSource, cv2.HOUGH_GRADIENT, dp=1, minDist=32, param1=100, param2=19,
                                    minRadius=20, maxRadius=28)
        if circles1 is not None:
            for circle1 in circles1[0]:
                x3, y3, r3 = int(circle1[0]), int(circle1[1]), int(circle1[2])
                center = (x3, y3)
                # r3 = 19

                # 画出圆的外接圆
                # cv2.circle(image1, (i[0], i[1]), i[2], (0, 255, 0), 5)

                # #画出圆心
                # cv2.circle(image1, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            logger.warning("在单张图片中没有检测到圆，请调整检测的圆半径，目前按照默认参数处理，请人工介入,确认按c,否决按q")
            cv2.imshow("detection", self.imgSource)
            # 检测图像的长和宽
            size = self.imgSource.shape
            w = size[1]  # 宽度
            h = size[0]  # 高度
            logger.info("要介入的图像的宽度为:{}".format(w))
            logger.info("要介入的图像的高度为:{}".format(h))
            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):
                logger.debug("人工介入成功,否决照片")
                self.isCheck = False
                cv2.destroyAllWindows()
                return
            elif key & 0xFF == ord("c"):
                logger.debug("人工介入成功,保留照片")
                cv2.destroyAllWindows()
            r3 = int(w/2)
            center = (int(w/2), int(h/2))
        cv2.circle(mask, center, r3, (255, 255, 255), thickness=cv2.FILLED)
        cv2.circle(mask_inverted, center, r3, (0, 0, 0), thickness=cv2.FILLED)

        image2 = cv2.bitwise_and(self.imgSource, self.imgSource, mask=mask)
        image2 = cv2.add(image2, mask_inverted)

        # 将不需要的区域设为白色
        image2[mask_inverted == 255] = 255

        # print(image2.shape)
        piece = cv2.resize(image2, (62, 62))

        piece[_mask] = 255
        self.piece = piece

        # self.piece = cv2.GaussianBlur(piece, (5, 5), 0)
        # piece = np.where(piece > 70, 255, 0)

        # 将灰度图像进行二值化处理
        # _, self.binary_image = cv2.threshold(self.piece, 70, 255, cv2.THRESH_BINARY)

    def __call__(self):
        if self.isCheck:
            return self.piece
        else:
            return False


class ChessImageProcessing:
    def __init__(self, image):
        # 转为灰度图
        self.num = 0
        self.chess_recognize_finish = None
        self.image = np.array(image).copy()

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # 中值滤波
        img_median = cv2.medianBlur(gray, 3)

        # 直方图均衡化
        img_equalize_hist = cv2.equalizeHist(img_median)

        self.img_equalize_hist = img_equalize_hist

    def __call__(self):
        # 返回处理的灰度图
        return self.img_equalize_hist

    def detection_circles(self, err=50, is_matching_coordinates=True):
        chess_recognize_finish = {}
        circles = cv2.HoughCircles(self.img_equalize_hist, cv2.HOUGH_GRADIENT, dp=1, minDist=38,
                                   param1=hough_circles_parameter_dict["param1"],
                                   param2=hough_circles_parameter_dict["param2"],
                                   minRadius=hough_circles_parameter_dict["r_min"],
                                   maxRadius=hough_circles_parameter_dict["r_max"])
        num = 0
        if circles is not None:
            for circles1 in tqdm(circles[0], desc="Identification circle move"):
                x, y, r = int(circles1[0]), int(circles1[1]), int(circles1[2])
                if not set_x_y_pixel_limit(x, y):
                    continue
                # r1 = 19
                if is_matching_coordinates:
                    for x1, y1, x2, y2 in chess_board_position_list:
                        if (math.sqrt((y2 - y) ** 2 + (x2 - x) ** 2)) <= err:
                            my_id = x1*10 + y1
                            chess_recognize_finish[my_id] = (x, y, r)
                num += 1
                # 画出圆的外接圆
                cv2.circle(self.image, (x, y), r, (0, 0, 255), 3)
                # 画出圆心
                # cv2.circle(image1, (x, y), 2, (0, 0, 255), 3)
        else:
            logger.warning("在整张图片中没有检测到圆，请调整检测的圆半径")
        self.num = num
        if is_matching_coordinates:
            self.chess_recognize_finish = chess_recognize_finish
            return self.chess_recognize_finish, num, self.image
        else:
            return num


def statistical_red_pixel(bgr_image):
    hsv1 = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    # 创建红色掩模
    mask1 = cv2.inRange(hsv1, lower_red, upper_red)
    # 由于红色是在色相空间上分布在两个值上（0和180），因此需要两个掩模来捕获红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv1, lower_red, upper_red)
    # # 合并两个红色掩模
    red_mask = mask1 + mask2
    red_pixels = cv2.countNonZero(red_mask)
    return red_pixels


def affirm_circle_num(picture_list):
    picture_num = len(picture_list)
    chess_num_list = []
    if picture_num == 0:
        logger.error("没有传入图片")
        return None, None
    else:
        for picture in picture_list:
            picture = ChessImageProcessing(picture)
            num = picture.detection_circles(is_matching_coordinates=False)
            chess_num_list.append(num)
        most_common = max(chess_num_list, key=chess_num_list.count)  # 求列表出现元素最多的数值
        logger.debug("识别的棋子个数列表：{}".format(chess_num_list))
        count = chess_num_list.count(most_common)
        # print(most_common)
        if count / picture_num >= 0.8:
            logger.info("结果可信，正确率为{}，大于等于80%".format(count / picture_num))

            index = chess_num_list.index(most_common)
            # print(index)
            return True, picture_list[index]
        else:
            logger.error("结果不可信，正确率为{},小于80%".format(count / picture_num))
            return False, None


def affirm_specifies_the_number_of_circles(frame_list=None, circle_num=None, number_of_continuous_circles_min=5):
    if frame_list is None:
        logger.error("Frames list is None")
    picture_num = len(frame_list)
    chess_num__index_list = []
    index = 0
    number_of_continuous_circles = 0
    if picture_num == 0:
        logger.error("没有传入图片")
        return None, None
    else:
        for picture in frame_list:
            current_red_pixel = statistical_red_pixel(picture)
            picture = ChessImageProcessing(picture)
            num = picture.detection_circles(is_matching_coordinates=False)
            if num == circle_num:
                number_of_continuous_circles = number_of_continuous_circles + 1
                chess_num__index_list.append(index)
            else:
                logger.warning("检测失败，继续检测")
                continue
            if number_of_continuous_circles >= number_of_continuous_circles_min:
                logger.info("检测成功")
                frame_list_new = [frame_list[index] for index in chess_num__index_list]
                return True, frame_list_new
            index = index + 1
            last_red_pixel = current_red_pixel
        return False, None


if __name__ == "__main__":
    img1 = cv2.imread(f"../output/photo_1.jpg")
    img2 = cv2.imread(f"../output/photo_2.jpg")
    # cv2.imshow("windows", red())
    cv2.waitKey()
