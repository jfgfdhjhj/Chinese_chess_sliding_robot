# coding=utf-8
# 用于标记的文本
chessTextAscii = ('B_King', 'B_Car', 'B_Hor', 'B_Elep', 'B_Bis', 'B_Canon', 'B_Pawn',
                  'R_King', 'R_Car', 'R_Hor', 'R_Elep', 'R_Bis', 'R_Canon', 'R_Pawn')

chessTextChi = ('将', '車', '馬', '炮', '士', '象', '卒', '帅', '車', '馬', '炮', '仕', '相', '兵')  # 棋子文本元组

B_King_none = -ord("k")
B_King = ord("k")

R_King_none = -ord("K")
R_King = ord("K")

B_Car_none = -ord("r")
B_Car = ord("r")

R_Car_none = -ord("R")
R_Car = ord("R")

B_Hor_none = -ord("n")
B_Hor = ord("n")

R_Hor_none = -ord("N")
R_Hor = ord("N")

B_Canon_none = -ord("c")
B_Canon = ord("c")

R_Canon_none = -ord("C")
R_Canon = ord("C")

B_Bis_none = -ord("a")
B_Bis = ord("a")

R_Bis_none = -ord("A")
R_Bis = ord("A")

B_Elep_none = -ord("b")
B_Elep = ord("b")

R_Elep_none = -ord("B")
R_Elep = ord("B")

B_Pawn_none = -ord("p")
B_Pawn = ord("p")

R_Pawn_none = -ord("P")
R_Pawn = ord("P")

ChessValidPos_ord = (
    ((B_King_none, B_King_none, B_King_none, B_King, B_King, B_King, B_King_none, B_King_none, B_King_none),  # 黑将合法位置
     (B_King_none, B_King_none, B_King_none, B_King, B_King, B_King, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King, B_King, B_King, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none),
     (B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none, B_King_none)),

    ((B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),  # 黑車合法位置,不用检测位置
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car),
     (B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car, B_Car)),

    ((B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),  # 黑马合法位置,不用检测位置
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor),
     (B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor, B_Hor)),

    ((B_Elep_none, B_Elep_none, B_Elep, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep, B_Elep_none, B_Elep_none),  # 黑象合法位置
     (B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none),
     (B_Elep, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep),
     (B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none),
     (B_Elep_none, B_Elep_none, B_Elep, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep, B_Elep_none, B_Elep_none),
     (B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none),
     (B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none),
     (B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none),
     (B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none),
     (B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none, B_Elep_none)),

    ((B_Bis_none, B_Bis_none, B_Bis_none, B_Bis, B_Bis_none, B_Bis, B_Bis_none, B_Bis_none, B_Bis_none),  # 黑士合法位置
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis, B_Bis_none, B_Bis, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none),
     (B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none, B_Bis_none)),

    ((B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),  # 黑炮合法位置,不用检测位置
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon),
     (B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon, B_Canon)),

    ((B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none),  # 黑卒合法位置
     (B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none),
     (B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none, B_Pawn_none),
     (B_Pawn, B_Pawn_none, B_Pawn, B_Pawn_none, B_Pawn, B_Pawn_none, B_Pawn, B_Pawn_none, B_Pawn),
     (B_Pawn, B_Pawn_none, B_Pawn, B_Pawn_none, B_Pawn, B_Pawn_none, B_Pawn, B_Pawn_none, B_Pawn),
     (B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn),
     (B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn),
     (B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn),
     (B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn),
     (B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn, B_Pawn)),

    ((R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none,
      R_King_none),  # 红帅合法位置
     (R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none,
      R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none,
      R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none,
      R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none,
      R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none,
      R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none, R_King_none,
      R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King, R_King, R_King, R_King_none, R_King_none, R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King, R_King, R_King, R_King_none, R_King_none, R_King_none),
     (R_King_none, R_King_none, R_King_none, R_King, R_King, R_King, R_King_none, R_King_none, R_King_none)),

    ((R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),  # 红車合法位置,不用检测位置
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car),
     (R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car, R_Car)),

    ((R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),  # 红马合法位置,不用检测位置
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor),
     (R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor, R_Hor)),

    ((R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none),  # 红相合法位置
     (R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none),
     (R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none),
     (R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none),
     (R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none),
     (R_Elep_none, R_Elep_none, R_Elep, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep, R_Elep_none, R_Elep_none),
     (R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none),
     (R_Elep, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep),
     (R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep_none),
     (R_Elep_none, R_Elep_none, R_Elep, R_Elep_none, R_Elep_none, R_Elep_none, R_Elep, R_Elep_none, R_Elep_none)),

    ((R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),  # 红仕合法位置
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis, R_Bis_none, R_Bis, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis, R_Bis_none, R_Bis_none, R_Bis_none, R_Bis_none),
     (R_Bis_none, R_Bis_none, R_Bis_none, R_Bis, R_Bis_none, R_Bis, R_Bis_none, R_Bis_none, R_Bis_none)),

    ((R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),  # 红炮合法位置,不用检测位置
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon),
     (R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon, R_Canon)),

    ((R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn),  # 红兵合法位置
     (R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn),
     (R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn),
     (R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn),
     (R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn, R_Pawn),
     (R_Pawn, R_Pawn_none, R_Pawn, R_Pawn_none, R_Pawn, R_Pawn_none, R_Pawn, R_Pawn_none, R_Pawn),
     (R_Pawn, R_Pawn_none, R_Pawn, R_Pawn_none, R_Pawn, R_Pawn_none, R_Pawn, R_Pawn_none, R_Pawn),
     (R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none),
     (R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none),
     (R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none, R_Pawn_none))

)









