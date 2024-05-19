import re
import time

from chess_dev.src.chess import Chess
from chess_dev.src.pikafishEngine import UCCIEngine
from chess_dev.src.engine import dirpath
# from PySide6 import QtCore, QtWidgets, QtGui
from PyQt5 import QtWidgets
from chess_dev.src.pyqt5board import BoardFrame
# from chess_dev.src.board import BoardFrame
from chess_utils.logger import logger
# import psutil
import sys
import python_chinese_chess_main.cchess as cchess
# from XQPy_main.XQlightPy.position import Position
# from XQPy_main.XQlightPy.search import Search
# from XQPy_main.XQlightPy.cchess import move2Iccs, Iccs2move
import json


file_chess = "Checkerboard_positioning.txt"
try:
    with open(file_chess, 'r', encoding='utf-8') as f:
        chess_dic = json.loads(f.read())
except FileNotFoundError:  # 抛出文件不存在异常
    print("转换后的象棋位置文件不存在")

uni_pieces = {4+8: '车', 3+8: '马', 2+8: '相', 1+8: '仕', 0+8: '帅', 6+8: '兵', 5+8: '炮',
              4+16: '俥', 3+16: '傌', 2+16: '象', 1+16: '士', 0+16: '将', 6+16: '卒', 5+16: '砲', 0: '．'}
# engine_address = "C:\\Users\\jhf12\\Documents\\graduation_project\\pikafish_230218_2fold\\pikafish-bmi2.exe"
# search_time_ms = 5000
# pos = Position()
# search = Search(pos, 16)


# fen_global = "position fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1\n"
# command = [engine_address, 'position fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
# w move g3g4 g6g5',
#            'go depth 20']
# rnbakabnr/9/1c4c2/p1p1p1p1p/9/6P2/P1P1P3P/1C5C1/9/RNBAKABNR w
# rnbakabnr/9/1c5c1/p1p1p1p1p/9/6P2/P1P1P3P/1C5C1/9/RNBAKABNR b


# class PikafishEngine:
#     def __init__(self, engine_path=engine_address):
#         self.fen_position_list = []
#         self.isready = None
#         self.bestmove = None
#         self.engine_process = subprocess.Popen(engine_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
#                                                shell=True, encoding="utf-8")
#         self.pid = self.engine_process.pid
#         self.proc_engine = psutil.Process(self.pid)
#         # print(self.pid)
#         # # 发送初始命令，例如uci，用于初始化引擎
#         # engine_process.stdin.write("uci\n")
#         # engine_process.stdin.flush()
#         self.first_step = True
#         self.move_list = []
#
#     def __engine_cmd(self, cmd):
#         self.engine_process.stdin.write(cmd + "\n")
#         self.engine_process.stdin.flush()
#
#     def __engine_read_bestmove(self):
#         while True:
#             line = self.engine_process.stdout.readline()
#             if line != "":
#                 # print(line)
#                 if line.find("bestmove") != -1:
#                     self.bestmove = line[9:13]
#                     print("best_move:{}".format(self.bestmove))
#                     break
#
#     def __engine_resume(self):
#         self.proc_engine.resume()
#         print("引擎恢复运行")
#
#     def __engine_suspend(self):
#         self.proc_engine.suspend()
#         print("引擎暂停运行")
#
#     def __win_or_lose_judgment(self):
#
#         return
#
#     def init_engine(self):
#         self.__engine_cmd("uci")
#         self.__engine_cmd("isready")
#         while True:
#             line = self.engine_process.stdout.readline()
#             if line != "":
#                 print(line)
#                 if line.find("uciok") != -1:
#                     print("初始化成功")
#                 if line.find("readyok") != -1:
#                     print("readyok")
#                     self.isready = True
#                     break
#         if self.isready:
#             print("引擎准备好了")
#         else:
#             self.isready = False
#             print("引擎没有准备好")
#
#     def run_engine(self, fen_position="r1bakabr1/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C1N2/9/RNBAKABR1",
#                    depth=20, is_stand_begin=True, is_user_red=True, user_move=None, ban_move=None, engine_move=None):
#         if not self.isready:
#             return None
#         self.__engine_resume()
#         if self.first_step:
#             self.first_step = False
#             if is_stand_begin:
#                 self.__engine_cmd("position startpos")
#             if is_user_red:
#                 self.__engine_cmd("position fen " + fen_position + "w - - 0 1")
#                 self.__engine_cmd("position fen " + fen_position + "w - - 0 1 moves " + user_move)
#                 self.move_list.append(user_move)
#             else:
#                 self.__engine_cmd("position fen " + fen_position + "b - - 0 1")
#
#         else:
#             self.move_list.append(engine_move)
#             self.move_list.append(user_move)
#             if is_user_red:
#                 self.__engine_cmd("position fen " + fen_position + "w - - 0 1 moves " + " ".join(self.move_list))
#             else:
#                 self.__engine_cmd("position fen " + fen_position + "b - - 0 1 moves " + " ".join(self.move_list))
#         if ban_move:
#             self.__engine_cmd("banmoves {}".format(ban_move))
#         self.__engine_cmd("go depth {}".format(depth))
#         self.__engine_read_bestmove()
#         self.__engine_suspend()
#         return self.bestmove
#
#     def close_engine(self):
#         self.__engine_resume()
#         self.__engine_cmd("quit")
#         self.engine_process.stdin.close()
#         self.engine_process.stdout.close()
#         self.engine_process.wait()
#         # print(self.proc_engine.status())
#         # self.proc_engine.kill()
#         print("引擎已经关闭")


class PikafishEngineBoard(cchess.Board):
    colour_list = ["b", "w"]
    STATE_ILLEGAL = 0
    STATE_DRAW = 1
    STATE_FINISH = 2
    STATE_NORMAL = 3

    def __init__(self, _fen="r1bakabr1/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C1N2/9/RNBAKABR1 w - - 0 1", callback=None):
        self.fen_list = None
        self._fen = _fen
        if _fen is None:
            super().__init__()
        else:
            super().__init__(_fen)
        self.is_finish = None
        self.state = None
        self.move_list = []
        self.callback = callback
        self.chess_num = self.count_total_pieces(_fen)
        print(self.unicode(axes=True, axes_type=0))

    def count_total_pieces(self, fen):
        total_pieces = 0
        fen_parts = fen.split(' ')[0]
        for row in fen_parts.split('/'):
            for char in row:
                if char.isalpha():
                    total_pieces += 1
        return total_pieces

    def move(self, move):
        is_eat = False
        pattern = r'^[a-z]\d[a-z]\d$'
        if not re.match(pattern, move):
            logger.warning("输入的字符不匹配")
            return self.STATE_ILLEGAL, None, None
        is_legal = self.is_legal(cchess.Move.from_uci(move))
        if is_legal:
            if self.is_capture(cchess.Move.from_uci(move)):
                is_eat = True
                self.chess_num = self.chess_num - 1
                self.move_list.append(move)

            self.push(cchess.Move.from_uci(move))
        if not self.move_list:
            fen_list = self._fen
        else:
            fen_list = self._fen + " moves " + " ".join(self.move_list)
        if not is_legal:
            self.state = self.STATE_ILLEGAL
            logger.warning("进行移动的字符不合法")
            return self.STATE_ILLEGAL, None, fen_list

        # 查找第一个空格的索引
        space_index = self.fen().find(' ')
        letter_after_space = self.fen()[space_index + 1]
        assert (letter_after_space in self.colour_list)
        colour_index = self.colour_list.index(letter_after_space)

        is_sixty_moves = self.is_sixty_moves()
        is_fourfold_repetition = self.is_fourfold_repetition()
        is_insufficient_material = self.is_insufficient_material()

        if is_sixty_moves or is_fourfold_repetition or is_insufficient_material:
            self.state = self.STATE_DRAW
            return self.STATE_DRAW, None, None

        is_checkmate = self.is_checkmate()
        is_stalemate = self.is_stalemate()
        is_perpetual_check = self.is_perpetual_check()
        is_game_over = self.is_game_over()
        if is_checkmate or is_stalemate or is_perpetual_check:
            self.is_finish = True

        if is_game_over or self.is_finish:
            self.state = self.STATE_FINISH
            return self.STATE_FINISH, None, self.colour_list[colour_index - 1]

        print(self.unicode(axes=True, axes_type=0))
        logger.debug(self.fen())
        self.state = self.STATE_NORMAL
        return self.STATE_NORMAL, is_eat, fen_list, self.is_check()

    # def victory_judgment(self):
    #     if self.state != self.STATE_NORMAL:
    #         return
    #     else:
    #         if self.state == self.STATE_DRAW:
    #             print()


# def print_board(pos):
#     print()
#     for i, row in enumerate(np.asarray(pos.squares).reshape(16, 16)[3:3+10, 3:3+9]):
#         print(' ', 9 - i, ''.join(uni_pieces.get(p, p) for p in row))
#     print('    ａｂｃｄｅｆｇｈｉ\n\n')


def main(fen=None):
    # _engine_address = dirpath / 'engines/Pikafish/pikafish-bmi2.exe'
    import system
    engine_address = system.engine_address
    app = QtWidgets.QApplication(sys.argv)
    ui = BoardFrame()
    engine = UCCIEngine(engine_address)
    # move_list = []

    def callback(move_type, data):
        # logger.warning(move_type)
        if move_type in (Chess.MOVE,):
            # if engine.sit.turn == Chess.RED:
            engine.position()
            engine.move(data[0][0], data[0][1])
            logger.info(data)
            # mylogger.warning(data[1])
            # mylogger.warning(engine.sit.fpos)
            # mylogger.warning(engine.sit.tpos)
            ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
            # move_list.append(data[1])

            res1 = board.move(data[1])
            logger.debug(res1)
            if res1[0] != board.STATE_NORMAL:
                if res1[0] == board.STATE_DRAW:
                    print("平局")
                elif res1[0] == board.STATE_FINISH:
                    if res1[2] == board.colour_list[0]:
                        print("黑色胜利")
                    else:
                        print("红色胜利")
                return
            else:
                if res1[3]:
                    print("将军")
            # board.push(cchess.Move.from_uci(data[1]))
            # print(board.unicode(axes=True, axes_type=0, invert_color=False))
            # mylogger.debug(engine.sit.turn)
            # result = engine.sit.move(data1[0], data1[1])
            while True:
                _move = input("输入走法:")
                # _move = "h7i7"
                # move_list.append(_move)
                res2 = board.move(_move)
                if res2[0] == board.STATE_ILLEGAL:
                    logger.warning("输入不合法,请重新输入")
                    continue
                else:
                    break
            logger.debug(res2)
            if res2[0] != board.STATE_NORMAL:
                if res2[0] == board.STATE_DRAW:
                    print("平局")
                elif res2[0] == board.STATE_FINISH:
                    if res2[3] == board.colour_list[0]:
                        print("黑色胜利")
                    else:
                        print("红色胜利")
                return
            else:
                if res2[3]:
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
            engine.go(depth=20)
        elif move_type == Chess.CHECKMATE:
            return
        # while engine.undo():
            # ui.board.setBoard(engine.board, engine.fpos, engine.tpos)
            # time.sleep(0.01)
    engine.callback = callback
    engine.start()
    time.sleep(1)
    board = PikafishEngineBoard(fen)
    engine.sit.parse_fen(fen)
    engine.position(fen)
    choice = input("你想要： \n\t1. 执红先行\n\t2. 执黑后行\n\t 请选择:\n")
    assert (choice in ["1", "2"])
    if choice == "1":
        while True:
            move = input("输入走法:")
            # _move = "h7i7"
            # move_list.append(move)
            res2 = board.move(move)
            if res2[0] == board.STATE_ILLEGAL:
                logger.warning("输入不合法,请重新输入")
                continue
            else:
                ui.show()
                break
        data3 = engine.sit.parse_move(move)
        # result = engine.sit.move(data1[0], data1[1])
        engine.move(data3[0], data3[1])
        ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
        engine.position()
        engine.go(depth=20)
    else:
        ui.show()
        engine.go(depth=20)
    app.exec()
    # thread.join()
    engine.close()

# turn_to_go = "w"
# fen = "r1bakabr1/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C1N2/9/RNBAKABR1 w - - 0 1"
# fen1 = "rnbakab1r/9/1c4nc1/p1p1p1p1p/9/8P/P1P1P1P2/1C5C1/9/RNBAKABNR w - - 2 2"
# fen3 = "3n1k3/4P2r1/6P1b/9/R8/2r6/9/3p4R/1nc1p1p2/3K5"
# fen3 = fen3 + f" {turn_to_go} - - 0 1"


if __name__ == "__main__":
    pass
    # main(fen=fen)
    # board = PikafishEngineBoard(fen)
    # print(board.fen())
    # fen = "r1bakabr1/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C1N2/9/RNBAKABR1 w - - 0 1"
    # # logging.basicConfig()
    # # logging.info("This is  INFO !!")
    # # # move2 = pos.move
    # best_move = None
    # engine = PikafishEngine()
    # engine.init_engine()
    # pos = FenPosition(fen)
    # first_flag = False
    # while True:
    #     user_move = input("输入移动:")
    #     pos_list_user = pos.move(str(user_move))
    #     if pos_list_user[0] == pos.STATE_ILLEGAL:
    #         print("走棋不合法")
    #         continue
    #     elif pos_list_user[0] == pos.STATE_FINISH:
    #         print("游戏结束，赢家是{}".format(pos_list_user[2]))
    #     elif pos_list_user[0] == pos.STATE_DRAW:
    #         print("判定和局")
    #     if not first_flag:
    #         first_flag = True
    #         best_move = engine.run_engine(user_move=str(user_move))
    #     else:
    #         best_move = engine.run_engine(user_move=str(user_move), engine_move=best_move)
    #     pos_list_engine = pos.move(best_move)
    #     if pos_list_engine[0] == 0:
    #         print("走棋不合法")
    #         continue
    #     elif pos_list_engine[0] == 2:
    #         print("游戏结束，赢家是{}".format(pos_list_engine[2]))
    #     elif pos_list_engine[0] == 1:
    #         print("判定和局")

    # chess_trans = CChessTrans(chess_dic)
    # print(chess_trans.trans_to_fen())
    # board = cchess.Board(chess_trans.trans_to_fen())
    # pos.fromFen(chess_trans.trans_to_fen() + "w - - 0 1")
    # choice = input("你想要： \n\t1. 执红先行\n\t2. 执黑后行\n\t 请选择:\n")
    # assert (choice in ["1", "2"])
    # mov = None
    # if choice == "2":
    #     mov = search.searchMain(64, search_time_ms)  # 搜索3秒钟
    #     pos.makeMove(mov)
    # while True:
    #     print_board(pos)
    #     # 人来下棋
    #     if mov:
    #         print("电脑的上一步：", move2Iccs(mov).replace("-", "").lower())
    #     hintmov = search.searchMain(64, 10)  # 搜索10毫秒，给出例子
    #     while True:
    #         user_step = input("请输入你的行棋步子，比如 " + move2Iccs(hintmov).replace("-", "").lower() + " \n" + \
    #                           "悔棋请输入 shameonme :\n").upper()
    #         if user_step == "shameonme".upper():
    #             mov = None
    #             pos.undoMakeMove()
    #             pos.undoMakeMove()
    #             break
    #         if len(user_step) == 4:
    #             user_step = user_step[:2] + "-" + user_step[2:]
    #         try:
    #             user_move = Iccs2move(user_step)
    #             assert (pos.legalMove(user_move))
    #         except:
    #             print("你的行棋不合法，请重新输入")
    #             continue
    #         pos.makeMove(user_move)
    #         break
    #
    #     winner = pos.winner()
    #     if winner is not None:
    #         if winner == 0:
    #             print("红方胜利！行棋结束")
    #         elif winner == 1:
    #             print("黑方胜利！行棋结束")
    #         elif winner == 2:
    #             print("和棋！行棋结束")
    #         break
    #
    #     if user_step != "shameonme".upper():
    #         # 电脑下棋
    #         mov = search.searchMain(64, search_time_ms)  # 搜索3秒钟
    #         pos.makeMove(mov)
    #
    #     winner = pos.winner()
    #     if winner is not None:
    #         if winner == 0:
    #             print("红方胜利！行棋结束")
    #         elif winner == 1:
    #             print("黑方胜利！行棋结束")
    #         elif winner == 2:
    #             print("和棋！行棋结束")
    #         break
