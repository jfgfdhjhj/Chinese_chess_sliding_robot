# coding=utf-8
"""
(C) Copyright 2021 Steven;
@author: Steven kangweibaby@163.com
基于原作者改造成pyqt5， @author：Silk
@date: 2024-4-5
PyQt5 棋盘基础控件，只用于棋盘的展示，和点击回调。
"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

from chess_dev.src.version import VERSION
import chess_dev.src.system as system

from chess_dev.src.chess import Chess
from chess_dev.src.engine import UCCIEngine
from chess_utils.logger import logger
import os
import ctypes
from PyQt5 import QtCore, QtWidgets, QtGui

dirpath = system.get_dirpath()


class BoardSignal(QtCore.QObject):
    refresh = QtCore.pyqtSignal()


class TwoCircleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("中国象棋")
        layout = QVBoxLayout()

        self.label = QLabel("你即将与机器对弈，请选择你执棋的颜色：")
        layout.addWidget(self.label)

        # 创建两个按钮作为选项
        self.button1 = QPushButton("1.用户执红|机器执黑")
        self.button1.clicked.connect(self.select_option1)
        layout.addWidget(self.button1)

        self.button2 = QPushButton("2.用户执黑|机器执红")
        self.button2.clicked.connect(self.select_option2)
        layout.addWidget(self.button2)

        self.setLayout(layout)

        self.selected_option = None
        # 设置对话框的位置和大小
        self.setGeometry(300, 500, 300, 100)  # 调整位置和大小
        # 将对话框设置为置顶
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

    def select_option1(self):
        self.selected_option = "1"
        self.accept()

    def select_option2(self):
        self.selected_option = "2"
        self.accept()


class Board(QtWidgets.QLabel):
    BOARD = os.path.join(dirpath, "images/board.png")
    MARK = os.path.join(dirpath, "images/mark.png")
    CHECK = os.path.join(dirpath, "images/check.png")
    FAVICON = os.path.join(dirpath, "images/black_bishop.png")

    IMAGES = {
        Chess.R: os.path.join(dirpath, "images/red_rook.png"),
        Chess.N: os.path.join(dirpath, "images/red_knight.png"),
        Chess.B: os.path.join(dirpath, "images/red_bishop.png"),
        Chess.A: os.path.join(dirpath, "images/red_advisor.png"),
        Chess.K: os.path.join(dirpath, "images/red_king.png"),
        Chess.C: os.path.join(dirpath, "images/red_cannon.png"),
        Chess.P: os.path.join(dirpath, "images/red_pawn.png"),
        Chess.r: os.path.join(dirpath, "images/black_rook.png"),
        Chess.n: os.path.join(dirpath, "images/black_knight.png"),
        Chess.b: os.path.join(dirpath, "images/black_bishop.png"),
        Chess.a: os.path.join(dirpath, "images/black_advisor.png"),
        Chess.k: os.path.join(dirpath, "images/black_king.png"),
        Chess.c: os.path.join(dirpath, "images/black_cannon.png"),
        Chess.p: os.path.join(dirpath, "images/black_pawn.png"),
    }

    ANIMATION_DURATION = 280

    def __init__(self, parent=None, callback=None):
        super().__init__(parent=parent)

        self.csize = 60
        self.board_image = QtGui.QPixmap(self.BOARD)

        if parent is None:
            self.setWindowIcon(QtGui.QIcon(self.FAVICON))
            self.setWindowTitle("Chinese Chess")

        app = QtWidgets.QApplication.instance()
        if app:
            # app.setWindowIcon(QtGui.QIcon(self.FAVICON))
            QtWidgets.QApplication.setWindowIcon(QtGui.QIcon(self.FAVICON))

        if os.name == 'nt':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                f'StevenBaby.Chess.{VERSION}'
            )

        self.setObjectName("Board")
        self.setScaledContents(True)

        self.animate = QtCore.QPropertyAnimation(self, b'geometry', self)

        self.resize(self.csize * Chess.W, self.csize * Chess.H)

        for chess, path in self.IMAGES.items():
            self.IMAGES[chess] = QtGui.QPixmap(path)

        mark = QtGui.QPixmap(self.MARK)

        self.mark1 = QtWidgets.QLabel(self)
        self.mark1.setPixmap(mark)
        self.mark1.setScaledContents(True)
        self.mark1.setVisible(False)

        self.mark2 = QtWidgets.QLabel(self)
        self.mark2.setPixmap(mark)
        self.mark2.setScaledContents(True)
        self.mark2.setVisible(False)

        check = QtGui.QPixmap(self.CHECK)
        self.mark3 = QtWidgets.QLabel(self)
        self.mark3.setPixmap(check)
        self.mark3.setScaledContents(True)
        self.mark3.setVisible(False)

        self.signal = BoardSignal()
        self.signal.refresh.connect(self.refresh)

        self.labels = np.zeros((Chess.W, Chess.H,), dtype=QtWidgets.QLabel)
        self.board = np.zeros((Chess.W, Chess.H,), dtype=int)

        self.fpos = None
        self.tpos = None
        self.check = None
        self.reverse = False

        self.update()

        self.callback = callback

    def setBoard(self, board, fpos=None, tpos=None):
        self.board = board
        self.fpos = fpos
        self.tpos = tpos
        self.signal.refresh.emit()

    def setReverse(self, reverse):
        self.reverse = reverse
        self.signal.refresh.emit()

    def setCheck(self, check):
        self.check = check
        self.signal.refresh.emit()

    def move(self, board, fpos, tpos, callback=None, animate=True):
        if not animate:
            if callable(callback):
                callback()
            return

        label = self.getLabel(fpos)
        if not label:
            return

        label.setVisible(True)
        label.raise_()

        if self.animate.state() != QtCore.QAbstractAnimation.State.Running:
            self.animate.setTargetObject(label)
            self.animate.setDuration(self.ANIMATION_DURATION)
            self.animate.setStartValue(QtCore.QRect(self.getChessGeometry(fpos)))
            self.animate.setEndValue(QtCore.QRect(self.getChessGeometry(tpos)))
            self.animate.start()

        if callable(callback):
            QtCore.QTimer.singleShot(self.ANIMATION_DURATION, callback)

    def refresh(self):
        self.setPixmap(self.board_image)
        for x in range(Chess.W):
            for y in range(Chess.H):
                pos = (x, y)
                self.setChess(pos, self.board[pos])

        if self.fpos:
            self.mark1.setGeometry(self.getChessGeometry(self.fpos))
            self.mark1.setVisible(True)
        else:
            self.mark1.setVisible(False)

        if self.tpos:
            self.mark2.setGeometry(self.getChessGeometry(self.tpos))
            self.mark2.setVisible(True)
        else:
            self.mark2.setVisible(False)

        if self.check:
            self.mark3.setGeometry(self.getChessGeometry(self.check))
            self.mark3.setVisible(True)
        else:
            self.mark3.setVisible(False)

        super().update()

    def resizeEvent(self, event):
        w = self.parentWidget().width()
        h = self.parentWidget().height()

        height = h
        width = h / Chess.H * Chess.W

        if width > w:
            width = w
            height = width / Chess.W * Chess.H

        width = int(width)
        height = int(height)

        x = (w - width) // 2
        y = (h - height) // 2
        self.setGeometry(x, y, width, height)

        self.csize = width // Chess.W

        self.refresh()

    def mousePressEvent(self, event):
        if event.buttons() != QtCore.Qt.LeftButton:

            return super().mousePressEvent(event)

        pos = self.getPosition(event)
        if not pos:
            return
        logger.debug("click %s", pos)
        self.clickPosition(pos)

    def clickPosition(self, pos):
        if callable(self.callback):
            self.callback(pos)

    def getLabel(self, pos):
        if not pos:
            return None
        label = self.labels[tuple(pos)]
        if not label:
            return None
        return label

    def setChess(self, pos, chess):
        label = self.labels[pos]
        if not label:
            label = QtWidgets.QLabel(self)
            label.pos = pos
            self.labels[pos] = label

        if not chess:
            label.setVisible(False)
            return

        image = self.IMAGES[chess]
        label.setPixmap(image)
        label.setScaledContents(True)
        label.setGeometry(self.getChessGeometry(pos))
        label.setVisible(True)

    def getChessGeometry(self, pos):
        pos = self.fitPosition(pos)
        return QtCore.QRect(
            pos[0] * self.csize,
            pos[1] * self.csize,
            self.csize,
            self.csize
        )

    def fitPosition(self, pos):
        if self.reverse:
            return Chess.W - pos[0] - 1, Chess.H - pos[1] - 1
        else:
            return pos

    def getPosition(self, event):
        x = event.position().x() // self.csize
        y = event.position().y() // self.csize

        if x < 0 or x >= Chess.W:
            return None
        if y < 0 or y >= Chess.H:
            return None

        pos = (int(x), int(y))
        return self.fitPosition(pos)


class BoardFrame(QtWidgets.QFrame):

    def __init__(self, parent=None, board_class=Board):
        super().__init__(parent)
        self.board = board_class(self)

        self.resize(self.board.size())
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)  # 设置窗口置顶

    def resizeEvent(self, event):
        self.board.resizeEvent(event)
        return super().resizeEvent(event)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = BoardFrame()
    fen = "rnbakabnr/9/1c5c1/9/9/9/9/1C5C1/9/RNBAKABNR w - - 0 1"
    engine_address = dirpath / 'engines/Pikafish/pikafish-bmi2.exe'
    engine = UCCIEngine(engine_address)
    engine.start()
    ui.board.setBoard(engine.sit.board, engine.sit.fpos, engine.sit.tpos)
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
