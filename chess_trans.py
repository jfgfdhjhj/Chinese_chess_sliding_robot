import math
import re

from camera import frame_grab
from chess_utils.image_processing import ChessImageProcessing, statistical_red_pixel, affirm_circle_num
from CharacterTable import ChessValidPos_ord, chessTextAscii
import numpy as np
from chess_utils.logger import logger
from system import standard_chess_board

# 对应的fen字典
chess_fen = {"No": "_",
             "B_King": "k", "B_Car": "r", "B_Hor": "n", "B_Canon": "c", "B_Bis": "a", "B_Elep": "b", "B_Pawn": "p",
             "R_King": "K", "R_Car": "R", "R_Hor": "N", "R_Canon": "C", "R_Bis": "A", "R_Elep": "B", "R_Pawn": "P",
             "Uk": "U"}

chess_fen_to_id_dict = {"_": "No",
                        "k": "B_King", "r": "B_Car", "n": "B_Hor", "c": "B_Canon", "a": "B_Bis", "b": "B_Elep", "p": "B_Pawn",
                        "K": "R_King", "R": "R_Car", "N": "R_Hor", "C": "R_Canon", "A": "R_Bis", "B": "R_Elep", "P": "R_Pawn",
                        "U": "Uk"
                        }


def _parse_fen(fen):
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


def fen_string_to_key_id_list(fen_string):
    start_row = 0
    start_col = 9
    fen_list = []
    num = 0
    board = _parse_fen(fen_string)
    print(board)
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] != ".":
                fen_list.append([start_row, start_col, chess_fen_to_id_dict[board[row][col]]])
                num = num + 1
            start_row = start_row + 1
        start_row = 0
        start_col = start_col - 1

    print(fen_list)
    return fen_list, num


def match_fen_to_standard_board(fen):
    fen_list = []
    end_keyid_list, end_chess_num = fen_string_to_key_id_list(fen)
    for row, col, keyid in end_keyid_list:
        for row1, col1, pixel_row, pixel_col in standard_chess_board:
            if row1 == row and col1 == col:
                fen_list.append([row, col1, pixel_row, pixel_col, keyid])
    return fen_list


def key_id_trans(key_id):
    x_num, y_num = int(key_id[0] / 10), int(key_id[0] % 10)
    return x_num, y_num


def num_trans_key_id(row, col):
    key_id = row * 10 + col
    return key_id


def key_id_move_trans_fen_move(before_key_id, after_key_id):
    """
    默认棋盘为横着的情况，规定红色棋子在左，黑色棋子在右
    id的命名规则为横行*10 + 纵列，横行取值范围为0-8，纵行的取值范围为0-9
    :param before_key_id:
    :param after_key_id:
    :return: 转换为fen列表的移动
    """
    if len(before_key_id) > 1 or len(after_key_id) > 1:
        return False
    else:
        x1num, y1num = key_id_trans(before_key_id)
        x2num, y2num = key_id_trans(after_key_id)
        x1str = chr(ord('a') + x1num)
        x2str = chr(ord('a') + x2num)
        move_str = str(x1str) + str(y1num) + str(x2str) + str(y2num)
        return move_str


def fen_move_trans_key_id(move_str):
    pattern = r'^[a-z]\d[a-z]\d$'
    if not re.match(pattern, move_str):
        return False
    else:
        x1num = ord(move_str[0]) - ord('a')
        y1num = int(move_str[1])
        x2num = ord(move_str[2]) - ord('a')
        y2num = int(move_str[3])
        before_key_id = x1num * 10 + y1num
        after_key_id = x2num * 10 + y2num
        if before_key_id == after_key_id:
            return False
        return before_key_id, after_key_id


class LegalToJudgePosition:
    def __init__(self, board_ascii):
        chess_list = []
        # char = chess_fen[chess_id]
        for piece in range(len(str(board_ascii))):
            value = str(board_ascii)[piece]
            chess_list.append(value)
        while " " in chess_list:
            chess_list.remove(" ")  # 删除所有' '
        chess_list.append("\n")
        self.chess_list = chess_list
        # print(chess_list)

    def judge_all_id_is_legal(self):
        for chess_id in chessTextAscii:
            my_list = []
            row_list = []
            char = chess_fen[chess_id]
            for value in self.chess_list:
                # print(value)
                if value == "\n":
                    my_list.append(row_list)
                    row_list = []
                # elif value == ".":
                # row_list.append(0)
                elif value == char:
                    row_list.append(ord(char))
                else:
                    row_list.append(0)
            # print(my_list)
            num = chessTextAscii.index(chess_id)
            a = np.array(ChessValidPos_ord[num])
            b = np.array(my_list)
            c = a + b
            # print(c)
            for row in c:
                if 0 in row:
                    logger.error("{}棋子摆放位置不合法（只检测了不合法的第一个棋子）".format(chess_id))
                    return False
        logger.info("棋子摆放均合法")
        return True


class CChessTrans:
    def __init__(self, chess_x_y):
        chess_pieces = []
        num_list = []
        self.chess_x_y = chess_x_y
        self.board = [["_"] * 9 for _ in range(10)]
        self.is_chess_position_same = True
        self.is_chess_pieces_num_right = True
        for i, j, name in self.chess_x_y:
            num_list.append(int(i) * 10 + int(j))
            chess_pieces.append(name)
        logger.debug(chess_pieces)
        if chess_pieces.count("R_King") > 1 or chess_pieces.count("B_King") > 1:
            logger.error("king数量不对,R_King:{},B_King:{}".format(chess_pieces.count("R_King"), chess_pieces.count("B_King")))
            self.is_chess_pieces_num_right = False
        elif chess_pieces.count("R_Car") > 2 or chess_pieces.count("B_Car") > 2:
            logger.error("car数量不对,R_Car:{},B_Car:{}".format(chess_pieces.count("R_Car"), chess_pieces.count("B_Car")))
            self.is_chess_pieces_num_right = False
        elif chess_pieces.count("R_Hor") > 2 or chess_pieces.count("B_Hor") > 2:
            logger.error("hor数量不对,R_Hor:{},B_Hor:{}".format(chess_pieces.count("R_Hor"), chess_pieces.count("B_Hor")))
            self.is_chess_pieces_num_right = False
        elif chess_pieces.count("R_Canon") > 2 or chess_pieces.count("B_Canon") > 2:
            logger.error("canon数量不对,R_Canon:{},B_Canon:{}".format(chess_pieces.count("R_Canon"), chess_pieces.count("B_Canon")))
            self.is_chess_pieces_num_right = False
        elif chess_pieces.count("R_Bis") > 2 or chess_pieces.count("B_Bis") > 2:
            logger.error("bis数量不对,R_Bis:{},B_Bis{}".format(chess_pieces.count("R_Bis"), chess_pieces.count("B_Bis")))
            self.is_chess_pieces_num_right = False
        elif chess_pieces.count("R_Elep") > 2 or chess_pieces.count("B_Elep") > 2:
            logger.error("elep数量不对,R_Elep:{},B_Elep:{}".format(chess_pieces.count("R_Elep"), chess_pieces.count("B_Elep")))
            self.is_chess_pieces_num_right = False
        elif chess_pieces.count("R_Pawn") > 5 or chess_pieces.count("B_Pawn") > 5:
            logger.error("pawn数量不对,R_Pawn:{},B_Pawn:{}".format(chess_pieces.count("R_Pawn"), chess_pieces.count("B_Pawn")))
            self.is_chess_pieces_num_right = False
        is_duplicate = len(num_list) != len(set(num_list))
        if is_duplicate:
            logger.error("存在棋子位置重叠情况，请检查")
        elif not self.is_chess_pieces_num_right:
            self.is_chess_position_same = False
            logger.error("棋子数量存在问题，请检查")
        else:
            self.is_chess_position_same = False
            logger.info("不存在棋子位置重叠情况，继续运行")
            logger.info("棋子数量没有问题，继续运行")

    def trans_to_fen(self):
        fen = []
        fen_dict = {}
        if self.is_chess_position_same:
            logger.error("无法转换为fen字典，棋盘中有坐标重复")
            return False
        elif not self.is_chess_pieces_num_right:
            logger.error("无法转换为fen字典，棋盘中棋子数量不对")
            return False
        else:
            for coord in self.chess_x_y:
                row = 9 - coord[1]  # 横坐标对应的行号
                col = coord[0]  # 纵坐标对应的列号
                piece_type = chess_fen[coord[2]]
                self.board[row][col] = piece_type
                # print(row, col, piece_type)
                # 将坐标和棋子类型添加到FEN字典中
                fen = '/'.join([''.join(row) for row in self.board])
            fen_parts = []
            empty_count = 0
            for char in fen:
                if char == '/':
                    if empty_count > 0:
                        fen_parts.append(str(empty_count))
                        empty_count = 0
                    fen_parts.append('/')
                elif char == '_':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_parts.append(str(empty_count))
                        empty_count = 0
                    fen_parts.append(char)
            if empty_count > 0:
                fen_parts.append(str(empty_count))
            return str(''.join(fen_parts))


class RecognizeMove:
    def __init__(self, img_before, img_after):
        self.suspect_chess_dict = None

        # 两种状态，行走或没有移动和吃子,判断失误为Err
        move_state_list = ("Shift", "Take", "Err")
        self.img_before = img_before
        self.img_after = img_after
        self.indeterminacy_position_list = None

        image1 = ChessImageProcessing(self.img_before)
        self.previous_gray = image1()
        img_before_dict, img_before_num,  image1_circle = image1.detection_circles()
        self.img_before_dict = img_before_dict
        self.image1_circle = image1_circle

        image2 = ChessImageProcessing(self.img_after)
        self.current_gray = image2()
        img_after_dict, img_after_num, image2_circle = image2.detection_circles()
        self.img_after_dict = img_after_dict
        self.image2_circle = image2_circle

        logger.info("走棋前棋子数量:{}".format(img_before_num))
        logger.info("走棋之后棋子数量:{}".format(img_after_num))
        if img_before_num == img_after_num:
            move_state = move_state_list[0]
            self.move_state = move_state
        elif img_before_num - img_after_num == 1:
            move_state = move_state_list[1]
            self.move_state = move_state
        else:
            move_state = move_state_list[2]
            self.move_state = move_state
            logger.error("前后照片棋子数目不对，请检查参数")
        logger.info("棋子移动状态：{},注意Shift也有可能是没有移动".format(move_state))

    def detecting_moving_coordinates(self, threshold=1000, err=5):
        suspect_chess_dict = {}
        suspect_list = []
        pixel_change_before = []
        pixel_change_after = []
        if self.move_state == "Shift":
            before_list = []
            after_list = []
            for key, value in self.img_before_dict.items():
                try:
                    x, y, r = self.img_after_dict[key]
                    if (math.sqrt((value[0] - x) ** 2 + (value[1] - y) ** 2)) >= err:
                        suspect_list.append(key)

                except KeyError:
                    x, y, r = self.img_before_dict[key]
                    before_list.append(key)
                    pixel_change_before.append((x, y))
            if len(before_list) == 0:
                self.move_state = "No_move"
                return "No_move"
            for key, value in self.img_after_dict.items():
                if key not in self.img_before_dict:
                    x, y, _ = self.img_after_dict[key]
                    after_list.append(key)
                    pixel_change_after.append((x, y))
            suspect_chess_dict["null"] = suspect_list
            suspect_chess_dict["before"] = before_list
            suspect_chess_dict["after"] = after_list
            fen_move = key_id_move_trans_fen_move(suspect_chess_dict["before"], suspect_chess_dict["after"])
        elif self.move_state == "Take":
            confirmation_list = []
            logger.debug("img_before_dict{}".format(self.img_before_dict))
            logger.debug("img_after_dict{}".format(self.img_after_dict))
            before_list = []
            for key, value in self.img_before_dict.items():
                try:
                    x, y, r = self.img_after_dict[key]
                    if (math.sqrt((value[0] - x) ** 2 + (value[1] - y) ** 2)) >= err:
                        logger.debug("像素坐标误差：{}".format(math.sqrt((value[0] - x) ** 2 + (value[1] - y) ** 2)))
                        suspect_list.append(key)
                except KeyError:
                    before_list.append(key)
                    x, y, r = self.img_before_dict[key]
                    pixel_change_before.append((x, y))
            suspect_chess_dict["before"] = before_list
            logger.debug("before_list:{}".format(before_list))
            suspect_chess_dict["null"] = suspect_list
            logger.debug("suspect_chess_list{}".format(suspect_chess_dict["null"]))
            for key_id in suspect_chess_dict["null"]:
                x1, y1, r1 = self.img_before_dict[key_id]
                x2, y2, r2 = self.img_after_dict[key_id]
                red_pixel_before = statistical_red_pixel(self.img_before[y1 - r1: y1 + r1, x1 - r1: x1 + r1])
                red_pixel_after = statistical_red_pixel(self.img_after[y2 - r2: y2 + r2, x2 - r2: x2 + r2])
                logger.debug("red_pixel_before:{}".format(red_pixel_before))
                logger.debug("red_pixel_after:{}".format(red_pixel_after))
                if abs(red_pixel_before - red_pixel_after) >= threshold:
                    confirmation_list.append(key_id)
                    pixel_change_after.append((x2, y2))
                    logger.debug("key_id{}".format(key_id))
            if len(confirmation_list) == 0:
                for key_id in self.img_after_dict.keys():
                    x1, y1, r1 = self.img_before_dict[key_id]
                    x2, y2, r2 = self.img_after_dict[key_id]
                    red_pixel_before = statistical_red_pixel(self.img_before[y1 - r1: y1 + r1, x1 - r1: x1 + r1])
                    red_pixel_after = statistical_red_pixel(self.img_after[y2 - r2: y2 + r2, x2 - r2: x2 + r2])
                    if abs(red_pixel_before - red_pixel_after) >= threshold:
                        confirmation_list.append(key_id)
                        pixel_change_after.append((x2, y2))
                        logger.debug("key_id：{}".format(key_id))
            fen_move = key_id_move_trans_fen_move(suspect_chess_dict["before"], confirmation_list)
        else:
            fen_move = "a0a0"
            logger.error("中国象棋棋子移动识别失败，请检查")
        # print(suspect_list)
        if not fen_move:
            fen_move = "a0a0"
            logger.error("识别列表棋子数量大于1，请检查")
        self.suspect_chess_dict = suspect_chess_dict
        if True and 0:
            return fen_move, self.suspect_chess_dict, self.image1_circle, self.image2_circle
        else:
            return fen_move, pixel_change_before, pixel_change_after


class RobotMovePlanning:
    def __init__(self, img_before, fen_move, is_eat):
        self.img_before = img_before
        self.fen_move = fen_move
        self.is_eat = is_eat

        image1 = ChessImageProcessing(self.img_before)
        self.previous_gray = image1()
        img_before_dict, img_before_num, image1_circle = image1.detection_circles()
        self.img_before_dict = img_before_dict
        self.image1_circle = image1_circle
        logger.info("机器走棋前棋子数量:{}".format(img_before_num))
        key_id = fen_move_trans_key_id(fen_move)
        if not key_id:
            return
        else:
            self.before_keyid = key_id[0]
            self.after_keyid = key_id[1]

    def get_move_pixel(self):
        before_pixel = self.img_before_dict[self.before_keyid]
        before_pixel = (before_pixel[0], before_pixel[1])
        if self.is_eat:
            after_pixel = self.img_before_dict[self.after_keyid]
            after_pixel = (after_pixel[0], after_pixel[1])

        else:
            after_pixel = standard_chess_board[self.after_keyid]
            after_pixel = (after_pixel[2], after_pixel[3])

        logger.info("之后的像素:{}".format(after_pixel))
        return before_pixel, after_pixel, self.is_eat


def recognition_circle_multiple(time=1, num_frames=25, is_use_saved_picture=False, frame_list=None):
    identification_rounds = 1
    while True:
        if is_use_saved_picture:
            frames = frame_list
        else:
            frames = frame_grab(num_frames=num_frames)
        res_circle = affirm_circle_num(frames)
        if res_circle[0]:
            image_ = res_circle[1]
            return True, image_
        else:
            if identification_rounds >= time:
                logger.error(f"检测失败,一共检测了{identification_rounds}轮")
                break
            else:
                logger.warning(f"检测失败, 继续检测，目前是第{identification_rounds}轮")
                identification_rounds += 1
                if is_use_saved_picture:
                    return False, None
    return False, None

# class end_game_arrangement


if __name__ == "__main__":
    image_before = recognition_circle_multiple(time=5, num_frames=10)[1]

    # chess_trans = CChessTrans(chess_dic)
    # print(chess_trans.trans_to_fen())
    # board = cchess.Board(chess_trans.trans_to_fen())
    # print(board.is_legal)
    # print(board.status())
    # print(board.is_king_line_of_sight())
    # python_chinese_chess_main.cchess.svg.board(board, squares=board.attacks(cchess.E4))
    # print(board.unicode(axes_type=0, axes=True))
    # fen_move = key_id_move_trans_fen_move(2, 57)
    # print(fen_move)

    key = input("移动棋子了吗?")
    image_after = recognition_circle_multiple(time=5, num_frames=10)[1]
    res = RecognizeMove(image_before, image_after)
    move = res.detecting_moving_coordinates()
    print("相机识别move:{}".format(move[0]))
    # key = fen_move_trans_key_id("b0b1")
    # print(key)
