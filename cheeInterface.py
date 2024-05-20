import sys
import time

import system
from PyQt5 import QtWidgets

from XQPy_main.XQlightPy.cchess import Iccs2move
# from ArmIK.route_planning import Route
from chess_dev.src.chess import Chess
from chess_dev.src.pikafishEngine import UCCIEngine
from chess_dev.src.pyqt5board import BoardFrame, TwoCircleDialog
from chess_engine_api import PikafishEngineBoard
from chess_utils.image_processing import affirm_specifies_the_number_of_circles

from chess_utils.logger import logger
from chess_trans import RecognizeMove, recognition_circle_multiple, RobotMovePlanning
from end_game_arrangement import EndGameRoute, recognize_chess_position
from situation_recognize import initial_game_judgment, re_detect_chess_position
from camera import frame_grab
from slide_ele_control import ChessRobotEle
from XQPy_main.XQlightPy.position import Position

depth_times = 20
frame_times = 5
num_id = 1
turn_to_go = "w"
# 是否只进行引擎测试
only_test_engine = False
# 是否进行残局摆放,True：进行摆放用户提供的fen字符串，
#               False：识别用户摆放的局面
ending = True

fen1 = "2bakab1r/9/2n1c1n2/p1p1p1p1p/c8/4P1PNP/P1P5C/R6r1/9/1NBAKAB2"
fen2 = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
fen3 = "3n1k3/4P2r1/6P1b/9/R8/2r6/9/3p4R/1nc1p1p2/3K5"

# fen = input("请输入残局fen字符串")
my_fen = fen3

if ending:
    chess_recognize_list = recognize_chess_position()
    slide_robot = EndGameRoute(my_fen, chess_recognize_list)
    slide_robot.set_home()
    logger.info("滑轨设置原点成功")
    time.sleep(2)
    slide_robot.run_game()

if only_test_engine:
    _fen = my_fen
else:
    if ending:
        _fen = my_fen
    else:
        _result, _fen, _board_unicode = initial_game_judgment(is_show_windows=True)
        if _result:
            logger.info("last: %s %s\n%s", _result, _fen, _board_unicode)
            print(_fen)
        else:
            logger.error("棋子初始摆放位置不合法")
            raise RuntimeError("棋子初始摆放位置不合法，程序终止")

# order = input("请选择先手后手，1为先手，2为后手")
# if order == "2":
#     turn_to_go = "b"

fen = _fen + f" {turn_to_go} - - 0 1"
engine_address = system.engine_address
app = QtWidgets.QApplication(sys.argv)
ui = BoardFrame()
engine = UCCIEngine(engine_address)
board = PikafishEngineBoard(_fen=fen)
Picture = system.SaveLoadPicture()
pos = Position()
pos.fromFen(fen)

time.sleep(2)
# """
if not ending:
    slide_robot = ChessRobotEle()
    # input("请设置原点")
    logger.info("滑轨设置原点成功")
    slide_robot.set_home()
# """
# route = Route()
_move = None


def callback(move_type, data):
    global _move
    behind_first_flag = None
    # logger.warning(move_type)
    if move_type in (Chess.MOVE,):
        # if engine.sit.turn == Chess.RED:q
        engine.position()
        engine.move(data[0][0], data[0][1])
        logger.info(data)
        res_robot = board.move(data[1])
        ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
        logger.debug("引擎结果：{}".format(res_robot))

        user_step = data[1].upper()[:2] + "-" + data[1].upper()[2:]
        user_move = Iccs2move(user_step)
        assert (pos.legalMove(user_move))
        pos.makeMove(user_move)
        winner = pos.winner()
        if winner is not None:
            if winner == 0:
                print("红方胜利！行棋结束")
            elif winner == 1:
                print("黑方胜利！行棋结束")
            elif winner == 2:
                print("和棋！行棋结束")
            return True

        if not only_test_engine:
            robot_image_before = Picture.load_after_board("1")

            # res_robot = recognition_circle_multiple(time=5)
            # if res_robot[0]:
            #     robot_image_before = res_robot[1]
            # Picture.save_before_board("robot_image_before", robot_image_before)

            robot_recognize = RobotMovePlanning(robot_image_before, data[1], res_robot[1])
            before_pixel, after_pixel, is_eat = robot_recognize.get_move_pixel()

            logger.debug("走之前的像素坐标：{}，走之后的像素坐标：{}，是否吃子：{}".format(before_pixel, after_pixel, is_eat))

            # input("请帮助机器移动棋子")
            slide_robot.move_pixel_to_another_fixed_point(before_pixel, after_pixel, data[1], is_eat=is_eat)

            # route.move_one_point_to_another_point(before_pixel, after_pixel, is_eat=is_eat)
            # 多拍几帧备用
            frame_list = frame_grab(frame_times)

            # frame_list = slide_robot.frames_list
            # print(board.chess_num)
            try_num1_max = 0
            try_num2_max = 0
            try_num1 = 0
            try_num2 = 0

            res_5_1 = affirm_specifies_the_number_of_circles(slide_robot.frames_list, board.chess_num, 2)
            if res_5_1[0]:
                logger.info("通过归位前3帧识别，可以走棋！")
                robot_image_after = res_5_1[1][0]
                try_num1_max = len(res_5_1[1])

            elif not res_5_1[0]:
                res_5_2 = affirm_specifies_the_number_of_circles(frame_list, board.chess_num, 3)
                if res_5_2[0]:
                    try_num2_max = len(res_5_2[1])
                    logger.info("通过归位后5帧识别，可以走棋！")
                    robot_image_after = res_5_2[1][0]
                    try_num2_max = try_num2_max
            else:
                logger.info("通过归位前3帧和后5帧识别失败,继续识别！")
            # mylogger.warning(data[1])
            # mylogger.warning(engine.sit.fpos)
            # mylogger.warning(engine.sit.tpos)

            # move_list.append(data[1])

            # board.push(cchess.Move.from_uci(data[1]))
            # print(board.unicode(axes=True, axes_type=0, invert_color=False))
            # mylogger.debug(engine.sit.turn)
            # result = engine.sit.move(data1[0], data1[1])
            try_num = 0
            try_num_max = try_num1_max + try_num2_max

            while True:
                if not only_test_engine:
                    logger.info("正在检查机器是否移动成功！")
                    # robot_image_before = Picture.load_before_board("robot_image_before")
                    # res_robot = recognition_circle_multiple(time=1, is_use_saved_picture=True,
                    #                                         frame_list=slide_robot.frames_list)
                    # if res_robot[0]:
                    try:
                        _res = RecognizeMove(robot_image_before, robot_image_after)
                        _move = _res.detecting_moving_coordinates()
                        logger.info("相机识别move:{}".format(_move[0]))
                        _move = str(_move[0])
                        if _move == data[1]:
                            logger.info("机器移动正确！")
                            break
                        else:
                            try_num = try_num + 1
                            if try_num < try_num_max:
                                if res_5_1[0]:
                                    try_num1 = try_num1 + 1
                                    robot_image_after = res_5_1[1][try_num1]
                                elif res_5_2[0]:
                                    try_num2 = try_num2 + 1
                                    robot_image_after = res_5_2[1][try_num2]
                                if try_num1 >= try_num1_max:
                                    res_5_1[0] = False
                                elif try_num2 >= try_num2_max:
                                    res_5_2[0] = False
                            else:
                                logger.warning("检测前后帧均失败，局面将会重新检测！")
                                break
                    except Exception as e:
                        logger.error('错误类型是', e.__class__.__name__)
                        logger.error('错误明细是', e)
                        logger.warning("机器识别错误，程序继续运行")
                        # logger.warning("机器处于调试阶段，不能实现自动校正，请用户手动摆放棋子")
                        break

        if res_robot[0] != board.STATE_NORMAL:
            # if res_robot[0] == board.STATE_DRAW:
            #     print("平局")
            # elif res_robot[0] == board.STATE_FINISH:
            #     if res_robot[2] == board.colour_list[0]:
            #         print("黑色胜利")
            #     else:
            #         print("红色胜利")
            # return
            pass
        else:
            if res_robot[3]:
                print("将军")
        if not only_test_engine:
            behind_first_flag = True
            try_num = 0

        while True:
            if not only_test_engine:
                if behind_first_flag:
                    image_before = robot_image_after
                    Picture.save_before_board("1", image_before)
                else:
                    image_before = Picture.load_before_board("1")
                input("请继续走棋")
            else:
                _move = input("输入走法:")

            if not only_test_engine:
                current_fen = board.fen()
                current_fen = str(current_fen)
                image_after = recognition_circle_multiple(time=5, num_frames=15)[1]
                Picture.save_after_board("1", image_after)
                try:
                    _res = RecognizeMove(image_before, image_after)
                    _move = _res.detecting_moving_coordinates()
                    logger.info("相机识别move:{}".format(_move[0]))
                    _move = str(_move[0])

                    if _move == "a0a0":
                        logger.warning("前后棋子变化移动方案失败，将重新检测局面！")
                        re_detect_time_num = 0
                        while True:
                            re_detect_time_num = re_detect_time_num + 1
                            try:
                                re_detect_res = re_detect_chess_position(current_fen)
                                if re_detect_res[0]:
                                    _move = str(re_detect_res[1])
                                    break
                            except Exception as e:
                                logger.warning("第{}轮重新检测局面开始".format(re_detect_time_num + 1))
                                logger.error('错误类型是', e.__class__.__name__)
                                logger.error('错误明细是', e)
                                if re_detect_time_num >= 3:
                                    logger.error("检测失败")
                                    break

                except Exception as e:
                    logger.error('错误类型是', e.__class__.__name__)
                    logger.error('错误明细是', e)
                    logger.warning("前后棋子变化移动方案失败，将重新检测局面！")
                    re_detect_time_num = 0
                    while True:
                        re_detect_time_num = re_detect_time_num + 1
                        try:
                            re_detect_res = re_detect_chess_position(current_fen)
                            if re_detect_res[0]:
                                _move = str(re_detect_res[1])
                                break
                        except Exception as e:
                            logger.warning("第{}轮重新检测局面开始".format(re_detect_time_num + 1))
                            logger.error('错误类型是', e.__class__.__name__)
                            logger.error('错误明细是', e)
                            if re_detect_time_num >= 3:
                                logger.error("重新检测失局面失败")
                                break

            res3 = board.move(_move)
            if res3[0] == board.STATE_ILLEGAL:
                logger.warning("移动不合法,请重新移动")
                if not only_test_engine:
                    behind_first_flag = False
                continue
            else:
                break

        logger.debug(res3)
        user_step = _move.upper()[:2] + "-" + _move.upper()[2:]
        user_move = Iccs2move(user_step)
        assert (pos.legalMove(user_move))
        pos.makeMove(user_move)
        winner = pos.winner()
        if winner is not None:
            if winner == 0:
                print("红方胜利！行棋结束")
            elif winner == 1:
                print("黑方胜利！行棋结束")
            elif winner == 2:
                print("和棋！行棋结束")
            return True
        if res3[0] != board.STATE_NORMAL:
            # if res3[0] == board.STATE_DRAW:
            #     print("平局")
            # elif res3[0] == board.STATE_FINISH:
            #     if res3[3] == board.colour_list[0]:
            #         print("黑色胜利")
            #     else:
            #         print("红色胜利")
            # return
            pass
        else:
            if res3[3]:
                print("将军")
        # board.push(cchess.Move.from_uci(_move))
        # print(board.unicode(axes=True, axes_type=0, invert_color=False))
        # fenlist = fen + " moves " + " ".join(move_list)
        # mylogger.debug(engine.sit.turn)
        data3 = engine.sit.parse_move(_move)
        # result = engine.sit.move(data1[0], data1[1])
        engine.move(data3[0], data3[1])
        ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
        engine.position()
        engine.go(depth=depth_times)
    elif move_type == Chess.CHECKMATE:

        return
    # while engine.undo():
    # ui.board.setBoard(engine.board, engine.fpos, engine.tpos)
    # time.sleep(0.01)


def main():
    global fen
    first_flag = None
    engine.callback = callback
    engine.start()

    engine.sit.parse_fen(fen)
    engine.position(fen)
    ui.show()
    ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
    # 创建对话框
    dialog = TwoCircleDialog()
    if dialog.exec_():
        choice = dialog.selected_option
        # 用户点击了确定按钮
        if choice == "1":
            # 处理用户选择的情况
            if not only_test_engine:
                first_flag = True
            while True:
                if not only_test_engine:
                    if first_flag:
                        image_before = recognition_circle_multiple(time=5)[1]
                        Picture.save_before_board("1", image_before)
                    else:
                        image_before = Picture.load_before_board("1")
                    input("请走棋：")
                else:
                    move = input("输入第一步走法:")

                if not only_test_engine:
                    image_after = recognition_circle_multiple(time=5)[1]
                    Picture.save_after_board("1", image_after)
                    _res = RecognizeMove(image_before, image_after)
                    move = _res.detecting_moving_coordinates()
                    logger.info("相机识别move:{}".format(move[0]))
                    move = str(move[0])
                res2 = board.move(move)
                if res2[0] == board.STATE_ILLEGAL:
                    logger.warning("输入不合法,请重新输入,请退回的初始局面，重新识别走法")
                    if not only_test_engine:
                        first_flag = False
                    continue
                else:
                    break

            user_step = move.upper()[:2] + "-" + move.upper()[2:]
            user_move = Iccs2move(user_step)
            assert (pos.legalMove(user_move))
            pos.makeMove(user_move)
            winner = pos.winner()
            if winner is not None:
                if winner == 0:
                    print("红方胜利！行棋结束")
                elif winner == 1:
                    print("黑方胜利！行棋结束")
                elif winner == 2:
                    print("和棋！行棋结束")
                return True
            if res2[0] != board.STATE_NORMAL:
                # if res2[0] == board.STATE_DRAW:
                #     print("平局")
                # elif res2[0] == board.STATE_FINISH:
                #     if res2[2] == board.colour_list[0]:
                #         print("黑色胜利")
                #     else:
                #         print("红色胜利")
                # return False
                pass
            else:
                if res2[3]:
                    print("将军")

            data3 = engine.sit.parse_move(move)
            # result = engine.sit.move(data1[0], data1[1])
            engine.move(data3[0], data3[1])
            ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
            engine.position()
            engine.go(depth=depth_times)
        else:
            image_after = recognition_circle_multiple(time=5)[1]
            Picture.save_after_board("1", image_after)
            engine.go(depth=depth_times)

    app.exec()
    engine.close()
    # sys.exit(app.exec())


if __name__ == "__main__":
    main()
