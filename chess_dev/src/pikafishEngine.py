# coding=utf-8
"""
(C) Copyright 2021 Steven;
@author: Steven kangweibaby@163.com
基于原作者针对皮卡鱼引擎uci协议进行略微修改， @author：Silk
@date: 2024-04-5
用于处理 UCI 引擎，以及未来可能自己实现引擎留接口
"""

import os
import sys
import subprocess
import threading
import time
import traceback
from pathlib import Path
import re
import queue
import copy

from chess_dev.src.utils import attrdict
from chess_dev.src.chess import Chess
from chess_dev.src.situation import Situation
import chess_dev.src.system as system
from chess_utils.logger import logger as logger
import python_chinese_chess_main.cchess as cchess
dirpath = system.get_dirpath()


class Queue(queue.Queue):

    """
    该队列用于处理人机交互

    首先一个 Engine 线程读取 UCI 输出的行直接入对列
    其次需要一个 解析线程 单独处理 UCI 引擎的输出，在人不干预的情况下一直读队列的输出。
    但是人与引擎交互的时候，是在另一个线程进行的，于是在人与引擎交互的时候需要将解析线程停下来。

    人交互的时候:
        1. 交互线程 (目前是主线程) 首先向队列 put 下面的 mark，然后进入 wait
        2. 当解析线程读到 mark 的时候，notify 通知交互线程读取队列，自己进入 wait 等待交互完成
        3. 交互线程完成之后，notify 通知解析线程继续解析。
    """

    mark = 'a3c63f6d-a880-4bde-a176-82af83bcf793'

    def __init__(self, maxsize=0):
        super().__init__(maxsize)
        self.condition = threading.Condition()

    def __enter__(self):
        self.condition.acquire()
        self.put(self.mark)
        self.condition.wait()

    def get(self, block=True, timeout=None):
        while True:
            line = super().get(block=block, timeout=timeout)
            if line != self.mark:
                return line
            with self.condition:
                self.condition.notify()
                self.condition.wait()
                continue

    def __exit__(self, exc_type, exc_value, traceback):
        self.condition.notify()
        self.condition.release()


class Engine(threading.Thread):

    ENGINE_BOOT = 0
    ENGINE_IDLE = 1
    ENGINE_BUSY = 2

    ENGINE_MOVE = 3
    ENGINE_INFO = 4

    NAME = '基础引擎'

    def __init__(self):
        super().__init__(daemon=True)
        self.name = 'EngineThread'
        self.state = self.ENGINE_BOOT

        self.sit = Situation()
        self.stack = [self.sit]

        self.index = 0
        self.running = False
        self.usemillisec = True
        self.millisec = 1

    def callback(self, type, data):
        pass

    @property
    def checkmate(self):
        return self.sit.result == Chess.CHECKMATE

    def setup(self):
        pass

    def close(self):
        pass

    def set_index(self, index):
        if index < 0:
            return
        if index >= len(self.stack):
            return
        # logger.debug(f"{index}, {len(self.stack)}")

        self.index = index
        self.sit = self.stack[self.index]

    def move(self, fpos, tpos):
        logger.info("move from %s to %s", fpos, tpos)
        # logger.debug('start move stack %s index %s', self.stack, self.index)

        nidx = self.index + 1
        if nidx < len(self.stack) and (self.stack[nidx].fpos, self.stack[nidx].tpos) == (fpos, tpos):
            logger.debug("forward hint %s", self.sit.format_move(fpos, tpos))
            self.sit = self.stack[nidx]
            self.index = nidx
            result = self.sit.result
            return self.sit.result
        else:
            self.stack = self.stack[:self.index + 1]

        sit = copy.deepcopy(self.sit)

        result = sit.move(fpos, tpos)
        sit.result = result

        if not result:
            return result

        if result != Chess.INVALID:
            self.stack.append(sit)
            self.sit = sit
            self.index += 1
        else:
            self.sit.check = sit.check

        # logger.debug('finish move stack %s index %s', self.stack, self.index)

        return result

    def undo(self):
        # 悔棋

        # logger.debug('start undo stack %s index %s', self.stack, self.index)
        if self.index == 0:
            return False

        self.index -= 1

        self.sit = self.stack[self.index]

        logger.debug(self.sit)
        # logger.debug('finish undo stack %s index %s', self.stack, self.index)

        return True

    def redo(self):
        nidx = self.index + 1
        if nidx >= len(self.stack):
            return False

        self.index += 1
        self.sit = self.stack[self.index]
        return True

    def position(self, fen=None):
        return

    def go(self, depth=None, nodes=None,
           time=None, movestogo=None, increment=None,
           opptime=None, oppmovestogo=None, oppincrement=None,
           draw=None, ponder=None):
        return


class PipeEngine(Engine):

    def __init__(self, filename: Path):
        super().__init__()

        self.stderr = None
        self.stdout = None
        self.stdin = None
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.pipe = None

        self.parser_thread = threading.Thread(
            target=self.parser,
            daemon=True,
            name="ParserThread"
        )

        # 引擎输出队列
        self.outlines = Queue()
        self.setup()

    def close(self):
        self.running = False
        if self.pipe:
            self.pipe.terminate()
        self.parser_thread.join(5)
        self.join(5)

    def parse_line(self, line: str):
        # 具体解析的代码，子类实现
        pass

    def run(self):
        # 引擎有两个线程，自己读取引擎的输出，按行 put 到队列
        # 解析线程 从队列读取输出，进行解析

        self.running = True
        self.parser_thread.start()

        while self.running:
            line = self.readline()
            self.outlines.put(line)

    def parser(self):
        # 解析线程

        self.running = True
        while self.running:
            line = self.outlines.get()
            try:
                self.parse_line(line)
            except Exception:
                logger.error(traceback.format_exc())

    def send_command(self, command):
        # 向引擎写入指令
        try:
            logger.info("COMMAND: %s", command)
            line = f'{command}\n'.encode('gbk')
            self.stdin.write(line)
            self.stdin.flush()
        except IOError as e:
            logger.error("send command error %s", e)
            logger.error(traceback.format_exc())

    def decode(self, line):
        try:
            return line.decode("gbk")
        except UnicodeDecodeError:
            return line.decode("utf8")

    def readline(self):
        # 从引擎标准输出读取一行

        while True:
            try:
                line = self.stdout.readline().strip()
                line = self.decode(line)
                if not line:
                    return ""
                logger.info("OUTPUT: %s", line)
                return line
            except Exception as e:
                logger.error(traceback.format_exc())
                continue

    def clear(self):
        # 清空引擎输出

        with self.outlines:
            while True:
                try:
                    self.outlines.get_nowait()
                except queue.Empty:
                    return

    def setup(self):
        # 初始化引擎

        logger.info("open pipe %s", self.filename)

        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self.pipe = subprocess.Popen(
            [str(self.filename)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.dirname),
            startupinfo=startupinfo
        )
        self.stdin = self.pipe.stdin
        self.stdout = self.pipe.stdout
        self.stderr = self.pipe.stderr

    def my_deal(self, move=None):
        # self.stdout = "bestmove " + move + " ponder a9a0"
        # with self.outlines:
        # self.outlines.put("info string NNUE evaluation using pikafish.nnue enabled")
        self.outlines.put("bestmove " + move + " ponder a9a0")


class UCCIEngine(PipeEngine):

    """https://www.xqbase.com/protocol/cchess_ucci.htm"""

    def __init__(self, filename, callback=None):
        self.ids = attrdict()
        self.options = attrdict()
        self.callback = callback
        super().__init__(filename)

    def setup(self):
        super().setup()
        self.send_command('uci')

        while self.running:
            self.send_command('uci')
            line = self.readline()
            if not line:
                continue
            try:
                self.parse_line(line)
            except Exception:
                logger.error(traceback.format_exc())
            if self.state == self.ENGINE_IDLE:
                break

    def close(self):
        self.send_command('quit')
        return super().close()

    def parse_option(self, message: str):
        message = re.sub(' +', ' ', message)
        items = message.split()
        option = attrdict()

        key = None
        for var in items:
            if var in {'type', 'min', 'max', 'var', 'default'}:
                key = var
                continue
            elif not key:
                continue
            elif key == 'var':
                option.setdefault('vars', [])
                option.vars.append(var)
                continue
            else:
                option[key] = var
            if key == 'default':
                option.value = var
            key = None
        return option

    def set_option(self, option, value):
        if option not in self.options:
            logger.warning("option %s not supported!!!", option)
            return
        self.options[option].value = value
        command = f'setoption {option} {value}'
        self.send_command(command)

    def setup_option(self, name, option):
        if name == 'usemillisec':
            if option.value == 'true':
                self.usemillisec = True
                self.millisec = 1
            else:
                self.usemillisec = False
                self.millisec = 1000

    def position(self, fen=None):
        if not fen:
            fen = self.sit.format_fen()

        mark = 'fen '
        if fen.startswith('startpos'):
            mark = ''

        command = f'position {mark}{fen}'

        self.send_command(command)

    def banmoves(self, moves: list):
        if not moves:
            return
        command = f'banmoves {" ".join(moves)}'
        self.send_command(command)

    def go(self, depth=None, nodes=None,
           time=None, movestogo=None, increment=None,
           opptime=None, oppmovestogo=None, oppincrement=None,
           draw=None, ponder=None):

        command = "go"
        if draw:
            command += ' draw'
        elif ponder:
            command += ' ponder'

        if depth:
            command += f' depth {depth}'
        elif nodes:
            command += f' nodes {depth}'
        elif time:
            time //= self.millisec
            command += f' time {time}'

            if increment:
                increment //= self.millisec
                command += f' increment {increment}'
            elif movestogo:
                command += f' movestogo {movestogo}'
            elif opptime:
                opptime //= self.millisec
                command += f' opptime {opptime}'
            if oppincrement:
                oppincrement //= self.millisec
                command += f' oppincrement {oppincrement}'
            elif oppmovestogo:
                command += f' oppmovestogo {oppmovestogo}'
        else:
            return

        self.send_command(command)
        self.state = self.ENGINE_BUSY

    def ponderhit(self, draw=False):
        var = ''
        if draw:
            var = 'draw'
        command = f"ponderhint {var}"
        self.send_command(command)

    def probe(self, fen, moves=None):
        if not moves:
            moves = []
        command = f'probe {fen} moves {" ".join(moves)}'
        self.send_command(command)

    def pophash(self, line):
        pass

    def stop(self):
        self.send_command('stop')

    def isready(self):
        self.clear()
        self.send_command("isready")
        with self.outlines:
            line = self.outlines.get()
            if line == 'readyok':
                return True
            return False

    def parse_line(self, line: str):
        items = line.split(maxsplit=1)
        if not items:
            return
        instruct = items[0]
        if instruct == 'bye':
            self.running = False

        if instruct in ['readyok', 'bye']:
            return

        if instruct == 'uciok':
            self.state = self.ENGINE_IDLE
            return

        if instruct in {'id', 'option', 'info', 'pophash', 'bestmove'}:
            tup = items[1].split(maxsplit=1)
            if instruct == 'id':
                self.ids[tup[0]] = tup[1]
                return
            if instruct == 'option':
                self.options[tup[0]] = self.parse_option(tup[1])
                self.setup_option(tup[0], self.options[tup[0]])
                return

            if instruct == 'info':
                if callable(self.callback):
                    self.callback(Chess.INFO, line)
                return
            if instruct == 'pophash':
                if callable(self.callback):
                    self.callback(Chess.POPHASH, line)
                return

        self.state = self.ENGINE_IDLE

        type = None
        data = None

        if instruct == 'bestmove':
            moves = items[1].split()
            type = Chess.MOVE
            data1 = self.sit.parse_move(moves[0])
            data2 = moves[0]
            data = [data1, data2]
            if len(items) > 2:
                if items[2] == 'draw':
                    type = Chess.DRAW
                if items[2] == 'resign':
                    type = Chess.RESIGN

        elif instruct == 'nobestmove':
            if self.sit.idle > 100:
                type = Chess.DRAW
            else:
                type = Chess.NOBESTMOVE

        elif instruct == "Pikafish":
            pass

        else:
            logger.warning(instruct)
            return

        if callable(self.callback):
            self.callback(type, data)


# fen = "rnbakabnr/9/1c5c1/9/9/9/9/1C5C1/9/RNBAKABNR w - - 0 1"
# fen = "r1bakabr1/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C1N2/9/RNBAKABR1 w - - 0 1"
fen = "3n1k3/4P2r1/6P1b/9/R8/2r6/9/3p4R/1nc1p1p2/3K5 w -- 0 1"

def main():
    from PyQt5 import QtWidgets
    from pyqt5board import BoardFrame
    global fen
    engine_address = dirpath / 'engines/Pikafish/pikafish-bmi2.exe'
    app = QtWidgets.QApplication(sys.argv)
    ui = BoardFrame()
    engine = UCCIEngine(engine_address)
    move_list = []

    def callback(move_type, data):
        global fen
        # logger.warning(move_type)
        if move_type in (Chess.MOVE,):
            if engine.sit.turn == Chess.RED:
                engine.position()
                engine.move(data[0][0], data[0][1])
                logger.info(data)
                # mylogger.warning(data[1])
                # mylogger.warning(engine.sit.fpos)
                # mylogger.warning(engine.sit.tpos)
                ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
                move_list.append(data[1])
                board.push(cchess.Move.from_uci(data[1]))
                print(board.unicode(axes=True, axes_type=0, invert_color=False))
                # mylogger.debug(engine.sit.turn)
                # result = engine.sit.move(data1[0], data1[1])
                _move = input("输入走法")
                # _move = "h7i7"
                move_list.append(_move)
                board.push(cchess.Move.from_uci(_move))
                print(board.unicode(axes=True, axes_type=0, invert_color=False))
                fenlist = fen + " moves " + " ".join(move_list)
                # mylogger.debug(engine.sit.turn)
                data3 = engine.sit.parse_move(_move)
                # result = engine.sit.move(data1[0], data1[1])
                engine.move(data3[0], data3[1])
                ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
                logger.debug(fenlist)
                engine.position()
                engine.go(depth=20)
        # elif move_type == Chess.CHECKMATE:
        #     return
        #     while engine.undo():
        #         ui.board.setBoard(engine.sitboard, engine.fpos, engine.tpos)
        #         time.sleep(0.01)
        # elif move_type == Chess.INFO:
        #     print(move_type)
    engine.callback = callback
    engine.start()
    time.sleep(1)
    board = cchess.Board(fen)
    print(board.unicode(axes=True, axes_type=0, invert_color=False))
    engine.sit.parse_fen(fen)
    engine.position(fen)
    engine.go(depth=20)
    ui.show()
    app.exec()
    # thread.join()
    engine.close()


if __name__ == '__main__':
    main()
