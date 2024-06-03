#!/usr/bin/env python3
# encoding:utf-8
import time

import cv2
import json
import math
from tqdm import tqdm
import python_chinese_chess_main.cchess as cchess
from chess_trans import CChessTrans, LegalToJudgePosition, recognition_circle_multiple
from chess_utils.image_processing import ImageToFont, statistical_red_pixel
from chess_learn_train.train import single_chess_black_recognize, single_chess_red_recognize
from chess_utils.logger import logger
from system import hough_circles_parameter_dict, set_x_y_pixel_limit

# 测试图片保存路径
TestPicturePath = 'TestPicture/'
OutPicturePath = "output/"
file = "board_position.txt"
file_chess = "Checkerboard_positioning.txt"

# 用于标记的文本
chessTextAscii = ('B_King', 'B_Car', 'B_Hor', 'B_Elep', 'B_Bis', 'B_Canon', 'B_Pawn',
                  'R_King', 'R_Car', 'R_Hor', 'R_Elep', 'R_Bis', 'R_Canon', 'R_Pawn')

# 0--无棋子, 1--黑将, 2--黑車, 3--黑馬, 4--黑象, 5--黑士,  6--黑炮,  7--黑卒,
# 8--红帅,  9--红車, 10--红马, 11--红相, 12--红士, 13--红炮, 14--红兵, 15--未知

try:
    with open(file, 'r', encoding='utf-8') as f:
        chess_board_dic = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    logger.warning("象棋位置文件不存在")

# 标准棋盘列表
standard_chess_board = chess_board_dic["chess_board"]


def match_chess_position(original_piece_position, standard_piece_position, error=35):
    """
    进行识别到的坐标和标准坐标的匹配
    Args:
        original_piece_position:识别到的棋子坐标列表
        standard_piece_position: 标准的棋盘格子坐标列表
        error: 两个坐标之间的距离相差多少即可认为棋子在坐标格子上

    Returns:匹配完成的棋子列表，棋子个数

    """
    chess_recognize_finish = []
    for p, l, chess_id in original_piece_position:
        for i, j, x1, y1 in standard_piece_position:
            # print((math.sqrt((l - y1) ** 2 + (p - x1) ** 2)))
            if (math.sqrt((l - y1) ** 2 + (p - x1) ** 2)) <= error:
                chess_recognize_finish.append([i, j, chess_id])
    length = len(chess_recognize_finish)
    return chess_recognize_finish, length


# 遍历所有测试图片
# for dirItem in os.listdir(TestPicturePath):
# original_image_cut
# 是否读取摄像头拍摄的照片 # is_write_picture = False
def initial_game_judgment(is_create_trackbar=False, is_write_picture=True, is_show_windows=False):
    """
    用于初始局面的识别
    Args:
        is_create_trackbar:是否创建滑动块
        is_write_picture: 是否保存识别后的图片
        is_show_windows: 进行识别之后是否显示识别结果

    Returns: 如果识别完毕，返回识别的合法性，返回fen字符，返回unicode列表

    """
    # 定义文本内容和位置
    org = (0, 70)  # 左上角位置
    fontScale = 3
    color = (0, 0, 0)  # 黑色
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    # ------------------------检测阈值滑动条--------------------------------------
    # # 是否创建滑动条
    # is_create_trackbar = False
    # 霍夫圆阈值检测参数, 红黑棋子分类阈值检测参数, 中心点累加器阈值参数
    trackbar_parameter_list = [91, 236, 19]
    if is_create_trackbar:
        # 霍夫圆检测阈值滑动条回调函数
        def hough_circles_callback(value):
            pass
            print("霍夫圆阈值检测参数{}".format(value))

        # 红黑棋子分类阈值滑动条回调函数
        def color_classify_callback(value):
            pass
            print("红黑棋子分类阈值检测参数{}".format(value))

        def hough_circles_param2_callback(value):
            pass
            print("中心点累加器阈值参数{}".format(value))

        cv2.namedWindow("slider", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("slider", 720, 480)

        # 创建霍夫圆检测阈值滑动条
        cv2.createTrackbar('HoughCircles', 'slider', trackbar_parameter_list[0], 255, hough_circles_callback)

        # 创建红黑棋子分类阈值滑动条
        cv2.createTrackbar('ColorClassify', 'slider', trackbar_parameter_list[1], 2500, color_classify_callback)

        # 创建中心点累加器阈值滑动条
        cv2.createTrackbar('HoughCircles_param2', 'slider', trackbar_parameter_list[2], 255,
                           hough_circles_param2_callback)

    # ----------------------------------end-----------------------------

    picture_mun = 0
    while True and 1:
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        recognize_time_list = []
        # 用于统计棋子的列表
        chess_recognize = []
        # 变量清零
        blank_num = 0
        red_num = 0
        # 总棋子数量
        ChessNumCount = 0
        if is_create_trackbar:
            a = cv2.getTrackbarPos('HoughCircles', "slider")
            b = cv2.getTrackbarPos('ColorClassify', "slider")
            c = cv2.getTrackbarPos('HoughCircles_param2', "slider")
        else:
            a, b, c, = trackbar_parameter_list[0], trackbar_parameter_list[1], trackbar_parameter_list[2]

        if is_write_picture:
            if cap.isOpened():
                # flag, image = cap.read()
                image = recognition_circle_multiple(time=5)[1]
                cv2.imwrite(r"C:\Users\jhf12\Documents\graduation_project\chess_robot\photos\original_image_cut.png",
                            image)
            else:
                logger.warning("摄像头未开启")
                break
        else:
            image = cv2.imread(r"C:\Users\jhf12\Documents\graduation_project\chess_robot\photos\original_image_cut.png")
            is_write_picture = 1
        imgSource = image.copy()
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 中值滤波
        imgMedian = cv2.medianBlur(gray, 3)

        # 直方图均衡化
        imgEqualizeHist = cv2.equalizeHist(imgMedian)

        # # 二值化
        # thresh, threshold_two = cv2.threshold(imgEqualizeHist, 100, 255, cv2.THRESH_BINARY)

        # 输入图像，方法（类型），dp(dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推)，
        # minDist：最短距离-可以分辨是两个圆否 则认为是同心圆 ,
        # param1 边缘检测时使用Canny算子的高阈值，
        # param2 中心点累加器阈值—候选圆心（霍夫空间内累加和大于该阈值的点就对应于圆心），
        # minRadius：检测到圆的最小半径，maxRadius：检测到圆的的最大半径
        # 用a代替param1, 用c代替param2 # 27
        circles = cv2.HoughCircles(
            imgEqualizeHist, cv2.HOUGH_GRADIENT, dp=1, minDist=38,
            param1=hough_circles_parameter_dict["param1"],
            param2=hough_circles_parameter_dict["param2"],
            minRadius=hough_circles_parameter_dict["r_min"],
            maxRadius=hough_circles_parameter_dict["r_max"]
        )
        # 检测到了圆
        if circles is not None:
            # 遍历每一个圆
            for circle in tqdm(circles[0], desc="Identification circle"):
                x, y, r1 = int(circle[0]), int(circle[1]), int(circle[2])
                if not set_x_y_pixel_limit(x, y):
                    continue
                # r1 = 19

                # 统计出红色像素的个数
                redPixelValueCount = statistical_red_pixel(imgSource[y - r1:y + r1, x - r1:x + r1])

                # -------------------------------------------------------------------------------------------------------

                # 将图片转换为灰度图
                image1_second = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)

                # 裁剪出指定的棋子部分
                image1 = image1_second[y - r1:y + r1, x - r1:x + r1]

                # 将裁剪的棋子图片进一步处理
                image_to_font = ImageToFont(image1)

                if image_to_font.isCheck:
                    image_font = image_to_font()
                else:
                    continue
                cv2.imwrite(OutPicturePath + "cut/after_processing/" + "ccut" +
                            str(ChessNumCount) + ".png", image_font)
                # ------------------------------------------------------------------------------------------------------

                # 红色像素个数用b代替下面的常数
                if redPixelValueCount > b:
                    # 0--将, 1--红車, 2--红馬, 3--红相, 4--红仕,  5--红炮,  6--红兵
                    Red_classes = ('R_King', 'R_Car', 'R_Hor', 'R_Elep', 'R_Bis', 'R_Canon', 'R_Pawn')
                    # 红色棋子用绿点标记
                    cv2.circle(imgSource, (x, y), 4, (0, 255, 0), -1)
                    # 标记检测到的棋子位置
                    cv2.circle(imgSource, (x, y), r1, (0, 0, 255), 2)
                    # 送入模型进行检测
                    prediction = single_chess_red_recognize(image_font)
                    # 得出棋子的ID
                    imgID = Red_classes[prediction]
                    red_num += 1
                    # 标记棋子ID
                    cv2.putText(imgSource, imgID, (x - r1, y - r1), font, 0.7, (0, 0, 255), 2)

                # 黑色
                else:
                    # 黑色棋子用白点标记
                    cv2.circle(imgSource, (x, y), 4, (255, 255, 255), -1)
                    # 0--黑将, 1--黑車, 2--黑馬, 3--黑象, 4--黑士,  5--黑炮,  6--黑卒
                    Black_classes = ('B_King', 'B_Car', 'B_Hor', 'B_Elep', 'B_Bis', 'B_Canon', 'B_Pawn')
                    # 送入模型进行检测
                    prediction = single_chess_black_recognize(image_font)
                    # 得出棋子的ID
                    imgID = Black_classes[prediction]
                    blank_num += 1
                    # 标记检测到的棋子位置
                    cv2.circle(imgSource, (x, y), r1, (0, 255, 0), 2)
                    # 标记棋子ID
                    cv2.putText(imgSource, imgID, (x-r1, y - r1), font, 0.7, (0, 0, 0), 2)

                # 增加对应的棋子到相应列表
                chess_recognize.append([x, y, imgID])

                # 存储棋子裁剪后的图片
                cv2.imwrite(OutPicturePath + "cut/gray_image/" + "cut" + str(ChessNumCount) + ".png",
                            gray[y - r1:y + r1, x - r1:x + r1])

                ChessNumCount += 1
            logger.info("黑色棋子个数:{}, 红色棋子个数:{}".format(blank_num, red_num))

            image_with_num = cv2.putText(imgSource, str(picture_mun), org, font, fontScale, color, thickness)
            if is_show_windows:
                cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow("window", 1920, 1080)
                cv2.imshow('window', image_with_num)
            cv2.imwrite(OutPicturePath + str(picture_mun) + ".image_after_identification.png", image_with_num)

            if ChessNumCount > 32:
                logger.error("初始局面检测棋子超过32个，请检查识别")
                result_judge_all_id_is_legal = False
                return result_judge_all_id_is_legal, None, None
            print(chess_recognize)

            chess_list, length1 = match_chess_position(chess_recognize, standard_chess_board, error=50)
            print(chess_list, length1)
            recognize_time_list.append(chess_recognize)

            with open(file_chess, 'w', encoding='utf-8') as f:
                f.write(json.dumps(chess_list))
            logger.info('配置文件写入完成!')
            chess_trans = CChessTrans(chess_list)
            fen = chess_trans.trans_to_fen()
            logger.debug((chess_trans.trans_to_fen()))
            board = cchess.Board(chess_trans.trans_to_fen())
            # print(board)
            print(board.unicode(axes=True, axes_type=0))
            board_unicode = board.unicode(axes=True, axes_type=0)
            legal = LegalToJudgePosition(board)
            result_judge_all_id_is_legal = legal.judge_all_id_is_legal()
            logger.info("局面检测合法性判断为{}".format(result_judge_all_id_is_legal))
        else:
            logger.error("在棋盘中没有检测到圆，请调整棋盘")
            fen = None
            result_judge_all_id_is_legal = None
            board_unicode = None
        print("请输入参数，n或者q:(n为继续检测，q为保存检测):")
        if is_show_windows:
            key = cv2.waitKey(0) & 0xFF  # 获取按键的ASCII码
        else:
            key = input()
        if is_show_windows:
            if key == ord('q'):
                # 人工核查棋子识别没有问题，按下q保存参数
                cv2.destroyAllWindows()
                cap.release()
                return result_judge_all_id_is_legal, fen, board_unicode
            elif key == ord('n'):
                # 如果人工核查出现识别错误的情况，按下n继续识别
                cv2.destroyAllWindows()
                picture_mun += 1
                continue
        else:
            if key == "q":
                # 人工核查棋子识别没有问题，按下q保存参数
                cap.release()
                return result_judge_all_id_is_legal, fen, board_unicode
            elif key == "n":
                # 如果人工核查出现识别错误的情况，按下n继续识别
                picture_mun += 1
                continue
        # if key == 'q':
        #     # print(external_contours)
        #     if is_show_windows:
        #         cv2.destroyAllWindows()
        #     cap.release()
        #     return result_judge_all_id_is_legal, fen, board_unicode
        # if key == 'n':
        #     picture_mun += 1
        #     if is_show_windows:
        #         cv2.destroyAllWindows()
        #     # print(external_contours)
        #     continue


def re_detect_chess_position(before_fen):
    """
    用于移动棋子的过程中出现报错时的再次识别，识别与上面的函数大同小异，只不过返回值返回的是
    移动的fen字符串
    Args:
        before_fen: 识别之前的fen列表

    Returns:移动的fen字符串

    """
    recognize_time_list = []
    # 用于统计棋子的列表
    chess_recognize = []
    # 变量清零
    blank_num = 0
    red_num = 0
    # 总棋子数量
    ChessNumCount = 0
    # 霍夫圆阈值检测参数, 红黑棋子分类阈值检测参数, 中心点累加器阈值参数
    trackbar_parameter_list = [91, 236, 19]
    a, b, c, = trackbar_parameter_list[0], trackbar_parameter_list[1], trackbar_parameter_list[2]
    image = recognition_circle_multiple(time=1)[1]
    imgSource = image.copy()
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 中值滤波
    imgMedian = cv2.medianBlur(gray, 3)

    # 直方图均衡化
    imgEqualizeHist = cv2.equalizeHist(imgMedian)

    # # 二值化
    # thresh, threshold_two = cv2.threshold(imgEqualizeHist, 100, 255, cv2.THRESH_BINARY)

    # 输入图像，方法（类型），dp(dp=1时表示霍夫空间与输入图像空间的大小一致，dp=2时霍夫空间是输入图像空间的一半，以此类推)，
    # minDist：最短距离-可以分辨是两个圆否 则认为是同心圆 ,
    # param1 边缘检测时使用Canny算子的高阈值，
    # param2 中心点累加器阈值—候选圆心（霍夫空间内累加和大于该阈值的点就对应于圆心），
    # minRadius：检测到圆的最小半径，maxRadius：检测到圆的的最大半径
    # 用a代替param1, 用c代替param2 # 27
    circles = cv2.HoughCircles(
        imgEqualizeHist, cv2.HOUGH_GRADIENT, dp=1, minDist=38,
        param1=hough_circles_parameter_dict["param1"],
        param2=hough_circles_parameter_dict["param2"],
        minRadius=hough_circles_parameter_dict["r_min"],
        maxRadius=hough_circles_parameter_dict["r_max"]
    )
    # 检测到了圆
    if circles is not None:
        # 遍历每一个圆
        for circle in tqdm(circles[0], desc="Identification circle"):
            x, y, r1 = int(circle[0]), int(circle[1]), int(circle[2])
            if not set_x_y_pixel_limit(x, y):
                continue
            # r1 = 19
            redPixelValueCount = statistical_red_pixel(imgSource[y - r1:y + r1, x - r1:x + r1])
            # -------------------------------------------------------------------------------------------------------
            image1_second = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)
            image1 = image1_second[y - r1:y + r1, x - r1:x + r1]
            image_to_font = ImageToFont(image1)
            if image_to_font.isCheck:
                image_font = image_to_font()
            else:
                continue
            # ------------------------------------------------------------------------------------------------------

            # 红色像素个数用b代替下面的常数
            if redPixelValueCount > b:
                # 0--红帅, 1--红車, 2--红馬, 3--红相, 4--红仕,  5--红炮,  6--红兵
                Red_classes = ('R_King', 'R_Car', 'R_Hor', 'R_Elep', 'R_Bis', 'R_Canon', 'R_Pawn')
                prediction = single_chess_red_recognize(image_font)
                imgID = Red_classes[prediction]
                red_num += 1

            # 黑色
            else:
                # 0--黑将, 1--黑車, 2--黑馬, 3--黑象, 4--黑士,  5--黑炮,  6--黑卒
                Black_classes = ('B_King', 'B_Car', 'B_Hor', 'B_Elep', 'B_Bis', 'B_Canon', 'B_Pawn')
                prediction = single_chess_black_recognize(image_font)
                imgID = Black_classes[prediction]
                blank_num += 1

            # 增加对应的棋子到相应列表
            chess_recognize.append([x, y, imgID])

            ChessNumCount += 1
        logger.info("黑色棋子个数:{}, 红色棋子个数:{}".format(blank_num, red_num))
        if ChessNumCount > 32:
            logger.error("初始局面检测棋子超过32个，请检查识别")
            result_judge_all_id_is_legal = False
            return result_judge_all_id_is_legal, None

        print(chess_recognize)
        chess_list, length1 = match_chess_position(chess_recognize, standard_chess_board, error=50)
        print(chess_list, length1)
        _chess_trans = CChessTrans(chess_list)
        after_fen = _chess_trans.trans_to_fen()
        logger.debug((_chess_trans.trans_to_fen()))
        _board = cchess.Board(_chess_trans.trans_to_fen())
        _legal = LegalToJudgePosition(_board)
        result_judge_all_id_is_legal = _legal.judge_all_id_is_legal()
        logger.info("局面检测合法性判断为{}".format(result_judge_all_id_is_legal))
        try:
            fen_move = two_fen_trans_move(before_fen, after_fen)
            return result_judge_all_id_is_legal, fen_move
        except Exception as e:
            result_judge_all_id_is_legal = False
            logger.error(e)
            return result_judge_all_id_is_legal, None


def _parse_fen(fen):
    """
    解析fen字符，将其转换为列表形式
    Args:
        fen:

    Returns:

    """
    board = []
    first_part = fen.split(' ')[0]
    fen_parts = first_part.split('/')
    # print(fen_parts)
    for row in fen_parts:
        board_row = []
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    board_row.append('.')
            else:
                board_row.append(char)
        board.append(board_row)
    return board


def find_move(initial_board, final_board):
    """
    根据前后的fen字符串判断棋子的走棋结果
    大致思路为排除法一一进行个判断，走棋只有三种结果：没有走棋，移动或者吃子，根据三种状况的不同一一排除
    Args:
        initial_board: 走之前的fen字符串
        final_board: 走之后的fen字符串

    Returns:

    """
    moving_pieces = []
    for row in range(len(initial_board)):
        for col in range(len(initial_board[row])):
            if initial_board[row][col] != final_board[row][col]:
                moving_pieces.append((row, col))
    if len(moving_pieces) == 0:
        return False

    x1 = moving_pieces[0][0]
    y1 = moving_pieces[0][1]

    x2 = moving_pieces[1][0]
    y2 = moving_pieces[1][1]
    chess_piece1 = initial_board[x1][y1]
    chess_piece2 = initial_board[x2][y2]

    chess_piece3 = final_board[x1][y1]
    chess_piece4 = final_board[x2][y2]
    if chess_piece1 != "." and chess_piece2 != ".":
        # 进行移动判断
        if chess_piece3 == ".":
            initial_board_list = (x1, y1)
            final_board_list = (x2, y2)
        else:
            initial_board_list = (x2, y2)
            final_board_list = (x1, y1)
    else:
        if chess_piece1 == chess_piece4 and chess_piece2 == chess_piece3:
            # 进行吃子判断
            if chess_piece3 == ".":
                initial_board_list = (x1, y1)
                final_board_list = (x2, y2)
            else:
                initial_board_list = (x2, y2)
                final_board_list = (x1, y1)
        else:
            # 根据排除法，状态就是没有移动，也没有吃子
            return False

    # print("initial_board_list", initial_board_list)
    # print("final_board_list", final_board_list)

    return initial_board_list, final_board_list


def convert_to_algebraic_notation(move):
    """
    将得到的move列表准换为fen_move字符
    Args:
        move:

    Returns:

    """
    letters = "abcdefghi"
    numbers = "0123456789"
    start_col, start_row = move[0]
    end_col, end_row = move[1]

    col_letter_start = letters[start_row]
    row_number_start = numbers[9 - start_col]
    col_letter_end = letters[end_row]
    row_number_end = numbers[9 - end_col]
    _fen_ = col_letter_start + row_number_start + col_letter_end + row_number_end
    return _fen_

# 两个FEN字符串
# fen1 = "R3kab2/4a4/2c1b4/C3p4/6p2/7rP/P3P1n2/3rBK1c1/N3N4/3A1AB1R b - - 1 1"
# fen2 = "R3kab2/4a4/2c1b4/C3p4/6p2/7r1/P3P1n1P/3rBK1c1/N3N4/3A1AB1R"


def two_fen_trans_move(before_fen, after_fen):
    # 解析FEN字符串并获取棋盘布局
    initial_board = _parse_fen(before_fen)
    final_board = _parse_fen(after_fen)

    # 找出移动的起始点和终止点
    move = find_move(initial_board, final_board)

    # 将起始点和终止点转换为象棋的移动表示形式
    algebraic_notation = convert_to_algebraic_notation(move)
    logger.info("再次识别的移动结果:{}".format(algebraic_notation))

    return algebraic_notation


if __name__ == "__main__":
    _result, _fen, _board_unicode = initial_game_judgment()
    if _result:
        print("last:", _result, _fen + "\n", _board_unicode)
    else:
        logger.error("棋子初始摆放位置不合法")
