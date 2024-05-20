import math
import os
import time
import cv2
from tqdm import tqdm
from chess_learn_train.train import single_chess_red_recognize, single_chess_black_recognize
from chess_trans import recognition_circle_multiple, match_fen_to_standard_board
from chess_utils.image_processing import statistical_red_pixel, ImageToFont
from chess_utils.logger import logger
from slide_ele_control import ChessRobotEle
from slideway.mapping import dar_x, dar_y, offset_y, HandInEyeCalibrationSlide
from system import hough_circles_parameter_dict, set_x_y_pixel_limit, chess_board_dic

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)

# # 构建要加载的文件的路径
# checkerboard_12_pixels_position_path = os.path.join(current_dir, "photos", "end_game_arrangement.png")
# 标准棋盘列表
standard_chess_board = chess_board_dic["chess_board"]


def recognize_chess_position():
    # 定义文本内容和位置
    org = (0, 70)  # 左上角位置
    fontScale = 3
    color = (0, 0, 0)  # 黑色
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    OutPicturePath = "output/"
    red_pigment_threshold = hough_circles_parameter_dict["color"]
    picture_mun = 0
    while True:
        # 用于统计棋子的列表
        chess_recognize_list = []
        # 变量清零
        blank_num = 0
        red_num = 0
        # 总棋子数量
        ChessNumCount = 0
        image = recognition_circle_multiple(time=5)[1]
        cv2.imwrite(r"C:/Users/jhf12/Documents/graduation_project/chess_robot/photos/end_game_arrangement.png", image)
        imgSource = image.copy()
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 中值滤波
        imgMedian = cv2.medianBlur(gray, 3)

        # 直方图均衡化
        imgEqualizeHist = cv2.equalizeHist(imgMedian)

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
                cv2.imwrite(OutPicturePath + "end_game_arrangement_cut/after_processing/" + "ccut" +
                            str(ChessNumCount) + ".png", image_font)
                # ------------------------------------------------------------------------------------------------------

                # 红色像素个数用b代替下面的常数
                if redPixelValueCount > red_pigment_threshold:
                    # 0--将, 1--黑車, 2--黑馬, 3--黑象, 4--黑士,  5--黑炮,  6--黑卒
                    Red_classes = ('R_King', 'R_Car', 'R_Hor', 'R_Elep', 'R_Bis', 'R_Canon', 'R_Pawn')
                    # 红色棋子用绿点标记
                    cv2.circle(imgSource, (x, y), 4, (0, 255, 0), -1)
                    # 标记检测到的棋子位置
                    cv2.circle(imgSource, (x, y), r1, (0, 0, 255), 2)
                    prediction = single_chess_red_recognize(image_font)
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
                    prediction = single_chess_black_recognize(image_font)
                    imgID = Black_classes[prediction]
                    blank_num += 1
                    # 标记检测到的棋子位置
                    cv2.circle(imgSource, (x, y), r1, (0, 255, 0), 2)
                    # 标记棋子ID
                    cv2.putText(imgSource, imgID, (x - r1, y - r1), font, 0.7, (0, 0, 0), 2)

                # 增加对应的棋子到相应列表
                chess_recognize_list.append([x, y, imgID])
                # 存储棋子裁剪后的图片，检查正确性
                cv2.imwrite(
                    OutPicturePath + "end_game_arrangement_cut/gray_image/" + "cut" + str(ChessNumCount) + ".png",
                    gray[y - r1:y + r1, x - r1:x + r1])

                ChessNumCount += 1
            logger.info("黑色棋子个数:{}, 红色棋子个数:{}".format(blank_num, red_num))
            image_with_num = cv2.putText(imgSource, str(picture_mun), org, font, fontScale, color, thickness)
            cv2.imwrite(OutPicturePath + str(picture_mun) + ".end_game_arrangement_image_after_identification.png",
                        image_with_num)

            if ChessNumCount > 32:
                logger.error("初始局面检测棋子超过32个，请检查识别")
                result_judge_all_id_is_legal = False
                # return result_judge_all_id_is_legal, None, None
        print(chess_recognize_list)
        cv2.imshow("windows", image_with_num)
        key = cv2.waitKey(0)
        if key == ord("q"):
            cv2.destroyAllWindows()
            return chess_recognize_list
        else:
            cv2.destroyAllWindows()
            picture_mun = picture_mun + 1
            continue


# chess_recognize_list = [[1038, 429, 'B_Hor'], [1056, 116, 'B_Canon'], [699, 139, 'B_Canon'], [458, 414, 'R_Hor'],
#                         [702, 645, 'R_Pawn'], [1063, 665, 'B_Pawn'], [1070, 849, 'R_Pawn'], [550, 742, 'R_Hor'],
#                         [571, 678, 'R_Elep'], [1039, 731, 'B_Bis'], [619, 749, 'R_Car'], [732, 708, 'R_Canon'],
#                         [1121, 543, 'B_Car'], [822, 458, 'R_Pawn'], [377, 390, 'R_Elep'], [454, 593, 'R_King'],
#                         [986, 527, 'B_Pawn'], [1065, 586, 'B_Elep'], [722, 573, 'B_King'], [980, 685, 'B_Pawn'],
#                         [1125, 618, 'B_Hor'], [1058, 518, 'B_Pawn'], [620, 380, 'R_Pawn'], [518, 632, 'R_Bis'],
#                         [541, 394, 'R_Car'], [961, 938, 'B_Car'], [641, 491, 'B_Bis'], [713, 471, 'B_Elep'],
#                         [727, 876, 'R_Pawn'], [1138, 467, 'B_Pawn'], [470, 525, 'R_Bis'], [571, 324, 'R_Canon']]

# first_coordinate_match_list, _ = match_chess_position(chess_recognize_list, standard_chess_board, error=100)
# print(first_coordinate_match_list)


class EndGameGenerate:
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

    temporarily_set_aside_list = []
    for z in range(num_points_z):
        for y in range(num_points_y):
            for x in range(num_points_x):
                point_index = z * num_points_y * num_points_x + y * num_points_x + x
                coordinate = [(x + 1) * delta_x, y * delta_y, z * delta_z]
                temporarily_set_aside_list.append(coordinate)

    STANDARD_GAME = 1
    END_GAMEE = 2
    err = 50
    err1 = 10

    def __init__(self, end_fen, recognize_list):

        self.is_piece_drop_count = False
        self.standard_res1 = None
        self.standard_res2 = None
        self.standard_res3 = None
        self.standard_res4 = None

        # 标准局面
        self.overlapping_pieces_list_plan2 = []
        self.waiting_end_keyid_list = []
        self.route_list2 = []
        self.overlapping_pieces_list_plan1 = []
        self.need_move_num = 0
        self.temporarily_set_aside_num = 0
        self.route_list1 = []
        self.no_need_overlapping_pieces_list = []
        self.need_need_overlapping_keyid_list = []
        self.need_overlapping_pieces_list = []
        self.two_col_piece_list = []
        self.no_need_overlapping_keyid_list = []
        self.slide_move_list = []
        # .............
        # 残局局面
        self.end_col_piece_list = []
        self.end_reserved_piece_list = None

        # .........
        self.game_mode = None

        self.end_fen = end_fen
        self.recognize_list = recognize_list
        self._end_keyid_list()
        self.hand_eye_slide = HandInEyeCalibrationSlide()

        if self.game_mode == self.STANDARD_GAME:
            self._standard_game_init()
        else:
            self._end_game_init()

    def mapping_pixel_list(self, pixel_list):
        slide_move_list = []
        for before_row_pixel, before_col_pixel, after_row_pixel, after_col_pixel in pixel_list:
            before_position_res = self.hand_eye_slide.get_points_slide_limit_x_y(before_row_pixel, before_col_pixel)
            after_position_res = self.hand_eye_slide.get_points_slide_limit_x_y(after_row_pixel, after_col_pixel)
            slide_move_list.append(
                [before_position_res[1], before_position_res[2], after_position_res[1], after_position_res[2]])

        return slide_move_list

    def mapping_home_move_list(self, home_move_list, is_slide_position_first=False):
        slide_move_list_home = []
        if is_slide_position_first:
            for temporarily_set_aside_list, pixel_row, pixel_col, chess_id in home_move_list:
                after_position_res = self.hand_eye_slide.get_points_slide_limit_x_y(pixel_row, pixel_col)
                slide_move_list_home.append([temporarily_set_aside_list, after_position_res[1], after_position_res[2]])
        else:
            for pixel_row, pixel_col, temporarily_set_aside_list, chess_id in home_move_list:
                before_position_res = self.hand_eye_slide.get_points_slide_limit_x_y(pixel_row, pixel_col)
                slide_move_list_home.append(
                    [before_position_res[1], before_position_res[2], temporarily_set_aside_list])
        return slide_move_list_home

    def _end_keyid_list(self):
        end_keyid_list = match_fen_to_standard_board(self.end_fen)
        if len(end_keyid_list) > len(end_keyid_list):
            self.game_mode = self.END_GAMEE
        elif len(end_keyid_list) == 32:
            self.game_mode = self.STANDARD_GAME
        self.end_keyid_list = end_keyid_list

    def eliminate_two_rows_of_coordinates(self):
        need_overlapping_pieces_list = []
        need_need_overlapping_keyid_list = []
        no_need_overlapping_pieces_list = []
        no_need_overlapping_keyid_list = []

        for row, col, pixel_row, pixel_col, end_chess_id in self.end_keyid_list:
            if [row, col] not in self.two_col_piece_list:
                need_overlapping_pieces_list.append([row, col, pixel_row, pixel_col, end_chess_id])
                need_need_overlapping_keyid_list.append(end_chess_id)
            else:
                no_need_overlapping_pieces_list.append([row, col, pixel_row, pixel_col, end_chess_id])
                no_need_overlapping_keyid_list.append(end_chess_id)
            self.waiting_end_keyid_list.append([pixel_row, pixel_col, end_chess_id])

        self.need_overlapping_pieces_list = need_overlapping_pieces_list
        self.need_need_overlapping_keyid_list = need_need_overlapping_keyid_list

        self.no_need_overlapping_pieces_list = no_need_overlapping_pieces_list
        self.no_need_overlapping_keyid_list = no_need_overlapping_keyid_list

    def screen_out_the_pieces_that_need_to_be_moved(self):
        overlapping_pieces_list = []
        for pixel_row1, pixel_col1, chess_id in self.recognize_list:
            for row2, col2, pixel_row2, pixel_col2, end_chess_id in self.need_overlapping_pieces_list:
                distance = math.sqrt((pixel_row1 - pixel_row2) ** 2 + (pixel_col1 - pixel_col2) ** 2)
                if distance <= self.err and chess_id in self.need_need_overlapping_keyid_list and chess_id != end_chess_id:
                    overlapping_pieces_list.append([pixel_row1, pixel_col1,
                                                    self.temporarily_set_aside_list[self.temporarily_set_aside_num],
                                                    chess_id])
                    self.temporarily_set_aside_num = self.temporarily_set_aside_num + 1
                    break
        for pixel_row1, pixel_col1, temporarily_set_aside_list, chess_id in overlapping_pieces_list:
            if [pixel_row1, pixel_col1, chess_id] in self.recognize_list:
                self.recognize_list.remove([pixel_row1, pixel_col1, chess_id])

        self.need_move_num = len(overlapping_pieces_list)
        self.overlapping_pieces_list_plan1 = overlapping_pieces_list
        # need_overlapping_pieces_list.remove([row2, col2, pixel_row2, pixel_col2, end_chess_id])
        print("overlapping_pieces_list_plan1:", self.overlapping_pieces_list_plan1)
        print("recognize_list1:, len:", self.recognize_list, len(self.recognize_list))
        return self.overlapping_pieces_list_plan1

    def standard_route_plan1(self):
        for row2, col2, pixel_row, pixel_col, end_chess_id in self.no_need_overlapping_pieces_list:
            for pixel_row1, pixel_col1, chess_id in self.recognize_list:
                if chess_id == end_chess_id:
                    self.route_list1.append([pixel_row1, pixel_col1, pixel_row, pixel_col])
                    self.recognize_list.remove([pixel_row1, pixel_col1, chess_id])
                    self.waiting_end_keyid_list.remove([pixel_row, pixel_col, end_chess_id])
                    break
        print("recognize_list2:, len:", self.recognize_list, len(self.recognize_list))
        print("route_list1:", self.route_list1)
        return self.route_list1

    def standard_route_plan2(self):
        print("self.need_overlapping_pieces_list", self.need_overlapping_pieces_list)
        print("self.recognize_list", self.recognize_list)
        cycle_index = len(self.need_overlapping_pieces_list)

        for i in range(cycle_index):
            for pixel_row1, pixel_col1, chess_id in self.recognize_list:
                other_mini_distance = 0
                mini_distance = 0
                flag = 0
                other_distance_list = []
                for row, col, pixel_row, pixel_col, end_chess_id in self.need_overlapping_pieces_list:
                    distance = math.sqrt((pixel_row - pixel_row1) ** 2 + (pixel_col - pixel_col1) ** 2)
                    if other_mini_distance <= distance:
                        other_mini_distance = distance
                        other_distance_list = [pixel_row1, pixel_col1, chess_id, distance]
                    if chess_id == end_chess_id:
                        if distance <= self.err1:
                            self.recognize_list.remove([pixel_row1, pixel_col1, chess_id])
                            self.waiting_end_keyid_list.remove([pixel_row, pixel_col, end_chess_id])
                            self.need_overlapping_pieces_list.remove([row, col, pixel_row, pixel_col, end_chess_id])
                            flag = 1
                            break
                        if mini_distance <= distance:
                            temporary_need_overlapping_pieces_list = [row, col, pixel_row, pixel_col, end_chess_id]
                            temporary_route_list2 = [pixel_row1, pixel_col1, pixel_row, pixel_col]
                            temporary_recognize_list = [pixel_row1, pixel_col1, chess_id]
                            temporary_waiting_end_keyid_list = [pixel_row, pixel_col, end_chess_id]
                            mini_distance = distance
                            flag = 2
                if flag == 2:
                    if other_distance_list[3] <= self.err1 and other_distance_list[2] != temporary_need_overlapping_pieces_list[4]:
                        logger.warning("存在棋子重叠情况,需要移动")
                    self.route_list2.append(temporary_route_list2)
                    self.recognize_list.remove(temporary_recognize_list)
                    self.waiting_end_keyid_list.remove(temporary_waiting_end_keyid_list)
                    self.need_overlapping_pieces_list.remove(temporary_need_overlapping_pieces_list)
                    break
                elif flag == 1:
                    break
                        # self.route_list2.append([pixel_row1, pixel_col1, pixel_row, pixel_col])
                        # self.recognize_list.remove([pixel_row1, pixel_col1, chess_id])
                        # self.waiting_end_keyid_list.remove([pixel_row, pixel_col, end_chess_id])
                        # break
        print("recognize_list3:, len:", self.recognize_list, len(self.recognize_list))
        print("route_list2:", self.route_list2)
        return self.route_list2

    def put_back_the_piece_that_needs_to_be_moved(self):
        overlapping_pieces_list2 = []
        for row, col, temporarily_set_aside_list, chess_id in self.overlapping_pieces_list_plan1:
            for pixel_row, pixel_col, end_id in self.waiting_end_keyid_list:
                if end_id == chess_id:
                    overlapping_pieces_list2.append([temporarily_set_aside_list, pixel_row, pixel_col, chess_id])
                    self.waiting_end_keyid_list.remove([pixel_row, pixel_col, end_id])
                    break
        self.overlapping_pieces_list_plan2 = overlapping_pieces_list2
        print("overlapping_pieces_list_plan2:", self.overlapping_pieces_list_plan2)
        print("waiting_end_keyid_list:", self.waiting_end_keyid_list)
        return self.overlapping_pieces_list_plan2

    def _standard_game_init(self):
        two_col_piece_list = []
        for i in range(9):
            x_num1, y_num1 = int(i * 10 / 10), int(i * 10 % 10)
            two_col_piece_list.append([x_num1, y_num1])
            x_num2, y_num2 = int((i * 10 + 9) / 10), int((i * 10 + 9) % 10)
            two_col_piece_list.append([x_num2, y_num2])
        two_col_piece_list.append([1, 2])
        two_col_piece_list.append([7, 2])
        two_col_piece_list.append([1, 7])
        two_col_piece_list.append([7, 7])

        self.two_col_piece_list = two_col_piece_list
        self.eliminate_two_rows_of_coordinates()

        res1 = self.screen_out_the_pieces_that_need_to_be_moved()
        res5 = self.mapping_home_move_list(res1, is_slide_position_first=False)
        res2 = self.standard_route_plan1()
        res6 = self.mapping_pixel_list(res2)
        res3 = self.standard_route_plan2()
        res7 = self.mapping_pixel_list(res3)
        res4 = self.put_back_the_piece_that_needs_to_be_moved()
        res8 = self.mapping_home_move_list(res4, is_slide_position_first=True)

        print("res5", res5)
        print("res6", res6)
        print("res7", res7)
        print("res8", res8)

        self.standard_res1 = res5
        self.standard_res2 = res6
        self.standard_res3 = res7
        self.standard_res4 = res8

    def end_move_unwanted_pieces(self):
        reserved_piece_list = []
        end_list_of_extra_pieces = []

        for row, col, pixel_row, pixel_col, end_chess_id in self.end_keyid_list:
            for pixel_row1, pixel_col1, chess_id in self.recognize_list:
                if end_chess_id == chess_id:
                    reserved_piece_list.append([pixel_row1, pixel_col1, chess_id])
                    self.recognize_list.remove([pixel_row1, pixel_col1, chess_id])
                    break

        for pixel_row, pixel_col, chess_id in self.recognize_list:
            end_list_of_extra_pieces.append([pixel_row, pixel_col,
                                             self.temporarily_set_aside_list[self.temporarily_set_aside_num], chess_id])
            self.temporarily_set_aside_num = self.temporarily_set_aside_num + 1
            if self.temporarily_set_aside_num >= 16:
                self.temporarily_set_aside_num = 0
                logger.warning("需要移除的棋子数量大于16")
                self.is_piece_drop_count = True

        self.recognize_list = reserved_piece_list[:]
        print("recognize_list,", self.recognize_list)
        print("end_list_of_extra_pieces:", end_list_of_extra_pieces)
        return end_list_of_extra_pieces

    def end_eliminate_six_col_of_coordinates(self):
        need_overlapping_pieces_list = []
        need_need_overlapping_keyid_list = []
        no_need_overlapping_pieces_list = []
        no_need_overlapping_keyid_list = []
        end_keyid_list = self.end_keyid_list[:]

        for row, col, pixel_row, pixel_col, end_chess_id in self.end_keyid_list:
            self.waiting_end_keyid_list.append([pixel_row, pixel_col, end_chess_id])
        index_num_time = len(self.end_keyid_list)
        for i in range(index_num_time):
            for row, col, pixel_row, pixel_col, end_chess_id in end_keyid_list:
                if [row, col] in self.end_col_piece_list:
                    no_need_overlapping_pieces_list.append([row, col, pixel_row, pixel_col, end_chess_id])
                    no_need_overlapping_keyid_list.append(end_chess_id)
                    end_keyid_list.remove([row, col, pixel_row, pixel_col, end_chess_id])
                    break

        for row, col, pixel_row, pixel_col, end_chess_id in end_keyid_list:
            need_overlapping_pieces_list.append([row, col, pixel_row, pixel_col, end_chess_id])
            need_need_overlapping_keyid_list.append(end_chess_id)

        self.need_overlapping_pieces_list = need_overlapping_pieces_list
        self.need_need_overlapping_keyid_list = need_need_overlapping_keyid_list

        self.no_need_overlapping_pieces_list = no_need_overlapping_pieces_list
        self.no_need_overlapping_keyid_list = no_need_overlapping_keyid_list

    def _end_game_init(self):
        end_col_piece_list = []
        for i in range(9):
            x_num1, y_num1 = int(i * 10 / 10), int(i * 10 % 10)
            end_col_piece_list.append([x_num1, y_num1])
            end_col_piece_list.append([x_num1, y_num1 + 1])
            end_col_piece_list.append([x_num1, y_num1 + 2])

            x_num2, y_num2 = int((i * 10 + 9) / 10), int((i * 10 + 9) % 10)

            end_col_piece_list.append([x_num2, y_num2 - 2])
            end_col_piece_list.append([x_num2, y_num2 - 1])
            end_col_piece_list.append([x_num2, y_num2])

        self.end_col_piece_list = end_col_piece_list
        res1 = self.end_move_unwanted_pieces()
        res6 = self.mapping_home_move_list(res1, is_slide_position_first=False)
        self.end_eliminate_six_col_of_coordinates()

        res2 = self.screen_out_the_pieces_that_need_to_be_moved()
        res7 = self.mapping_home_move_list(res2, is_slide_position_first=False)

        res3 = self.standard_route_plan1()
        res8 = self.mapping_pixel_list(res3)

        res4 = self.standard_route_plan2()
        res9 = self.mapping_pixel_list(res4)

        res5 = self.put_back_the_piece_that_needs_to_be_moved()
        res10 = self.mapping_home_move_list(res5, is_slide_position_first=True)

        print("res6", res6)
        print("res7", res7)
        print("res8", res8)
        print("res9", res9)
        print("res10", res10)

        self.end_res1 = res6
        self.end_res2 = res7
        self.end_res3 = res8
        self.end_res4 = res9
        self.end_res5 = res10

    def return_route_plan(self):
        if self.game_mode == self.STANDARD_GAME:
            return self.standard_res1, self.standard_res2, self.standard_res3, self.standard_res4
        elif self.game_mode == self.END_GAMEE:
            return self.end_res1, self.end_res2, self.end_res3, self.end_res4, self.end_res5

    def return_standard_res(self):
        return self.standard_res1, self.standard_res2, self.standard_res3, self.standard_res4

    def return_end_res(self):
        return self.end_res1, self.end_res2, self.end_res3, self.end_res4, self.end_res5


class EndGameRoute(ChessRobotEle):
    rising_height = 10

    def __init__(self, end_fen, _chess_recognize_list):
        super().__init__()
        self.end_game_generate = EndGameGenerate(end_fen, _chess_recognize_list)

    def move_group_home_list(self, move_list, is_slide_position_first=False, is_manual_piece_removal=False):
        num = 0
        if is_slide_position_first:
            for temporarily_set_aside_list, after_position_x, after_position_y in move_list:
                self.run_assigned_position(("x", int(temporarily_set_aside_list[0])),
                                           ("y", int(temporarily_set_aside_list[1])))
                self.run_assigned_position(("z", int(temporarily_set_aside_list[2])))
                self.fall_down(z=4)
                self.clint_ele.send_electromagnet_attract()
                time.sleep(0.2)
                self.uplift(self.rising_height)

                self.run_assigned_position(("x", int(after_position_x)), ("y", int(after_position_y)))
                self.fall_down(0)
                time.sleep(0.2)
                self.clint_ele.send_electromagnet_fall()
                time.sleep(0.2)
                self.uplift(0)

        else:
            for before_position_x, before_position_y, temporarily_set_aside_list in move_list:
                if is_manual_piece_removal:
                    num = num + 1
                self.run_assigned_position(("x", int(before_position_x)), ("y", int(before_position_y)))
                self.fall_down(z=4)
                self.clint_ele.send_electromagnet_attract()
                time.sleep(0.2)
                self.uplift(self.rising_height)
                self.run_assigned_position(("x", int(temporarily_set_aside_list[0])),
                                           ("y", int(temporarily_set_aside_list[1])))
                self.run_assigned_position(("z", int(temporarily_set_aside_list[2])))
                self.fall_down(0)
                time.sleep(0.2)
                self.clint_ele.send_electromagnet_fall()
                time.sleep(0.2)
                self.uplift(0)
                if is_manual_piece_removal:
                    if num >= 16:
                        input("请手动移除多余的棋子，移除完按回车")
                        num = 0

    def move_group_list(self, move_list):
        for before_position_x, before_position_y, after_position_x, after_position_y in move_list:
            self.run_assigned_position(("x", before_position_x), ("y", before_position_y))
            self.fall_down(z=4)
            self.clint_ele.send_electromagnet_attract()
            time.sleep(0.2)
            self.uplift(self.rising_height)

            self.run_assigned_position(("x", after_position_x), ("y", after_position_y))
            self.fall_down(0)
            time.sleep(0.5)
            self.clint_ele.send_electromagnet_fall()
            time.sleep(0.2)
            self.uplift(0)

    def run_game(self):
        if self.end_game_generate.game_mode == self.end_game_generate.STANDARD_GAME:
            self.run_standard_game()
        else:
            self.run_end_game()

    def run_standard_game(self):
        self.init()
        standard_res1, standard_res2, standard_res3, standard_res4 = self.end_game_generate.return_standard_res()

        self.move_group_home_list(standard_res1, is_slide_position_first=False)
        self.move_group_list(standard_res2)
        self.move_group_list(standard_res3)
        self.move_group_home_list(standard_res4, is_slide_position_first=True)
        self.return_to_home()

    def run_end_game(self):
        self.init()
        end_res1, end_res2, end_res3, end_res4, end_res5 = self.end_game_generate.return_end_res()

        self.move_group_home_list(end_res1, is_slide_position_first=False,
                                  is_manual_piece_removal=self.end_game_generate.is_piece_drop_count)
        self.move_group_home_list(end_res2, is_slide_position_first=False)
        self.move_group_list(end_res3)
        self.move_group_list(end_res4)
        self.move_group_home_list(end_res5, is_slide_position_first=True)
        self.return_to_home()
        input("请手动移除多余的棋子，防止影响后续对局，移除完按回车")


if __name__ == '__main__':

    fen1 = "2bakab1r/9/2n1c1n2/p1p1p1p1p/c8/4P1PNP/P1P5C/R6r1/9/1NBAKAB2"
    fen2 = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
    fen3 = "3n1k3/4P2r1/6P1b/9/R8/2r6/9/3p4R/1nc1p1p2/3K5 w"
    # fen = input("请输入残局fen字符串")
    chess_recognize_list = recognize_chess_position()
    end_game = EndGameRoute(fen3, chess_recognize_list)
    # end_game.set_home()
    # logger.info("滑轨设置原点成功")
    # time.sleep(2)
    # end_game.run_game()
