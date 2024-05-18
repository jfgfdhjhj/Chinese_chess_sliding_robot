import re


def parse_fen(fen):
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
    moving_pieces = []
    for row in range(len(initial_board)):
        for col in range(len(initial_board[row])):
            if initial_board[row][col] != final_board[row][col]:
                moving_pieces.append((row, col))

    x1 = moving_pieces[0][0]
    y1 = moving_pieces[0][1]

    x2 = moving_pieces[1][0]
    y2 = moving_pieces[1][1]
    chess_piece1 = initial_board[x1][y1]
    chess_piece2 = initial_board[x2][y2]

    chess_piece3 = final_board[x1][y1]
    chess_piece4 = final_board[x2][y2]
    if chess_piece1 != "." and chess_piece2 != ".":
        if chess_piece3 == ".":
            initial_board_list = (x1, y1)
            final_board_list = (x2, y2)
        else:
            initial_board_list = (x2, y2)
            final_board_list = (x1, y1)
    else:
        if chess_piece1 == chess_piece4 and chess_piece2 == chess_piece3:
            if chess_piece3 == ".":
                initial_board_list = (x1, y1)
                final_board_list = (x2, y2)
            else:
                initial_board_list = (x2, y2)
                final_board_list = (x1, y1)
        else:
            return False

    print("initial_board_list", initial_board_list)
    print("final_board_list", final_board_list)

    return initial_board_list, final_board_list

def convert_to_algebraic_notation(move):
    letters = 'abcdefghi'
    numbers = '0123456789'
    start_col, start_row = move[0]
    end_col, end_row = move[1]

    col_letter_start = letters[start_row]
    row_number_start = numbers[9-start_col]
    col_letter_end = letters[end_row]
    row_number_end = numbers[9-end_col]
    _fen_ = col_letter_start + row_number_start + col_letter_end + row_number_end
    _fen_ = _fen_.replace(" ", "")
    print(type(_fen_))
    print(_fen_)
    print(len(_fen_))
    return _fen_

# 两个FEN字符串
fen1 = "2bakab1r/9/2n1c1n2/p1p1p1p1p/c8/4P1PNP/P1P5C/R6r1/9/1NBAKAB2"
fen2 = "2bakab1r/9/2n1c1n2/p1p1p1p1p/c8/4P1PNP/P1P5C/7r1/9/RNBAKAB2 w - - 2 3"

# fen1 = "rnbakab1r/9/1c4nc1/p1p1p1p1p/9/8P/P1P1P1P2/1C3C3/9/RNBAKABNR"
# fen2 = "rnbakab1r/9/1c4nc1/p1p1p1p1p/9/8P/P1P1P1P2/1C5C1/9/RNBAKABNR w - - 2 2"

fen1 = "2bakab1r/9/1R2c1n2/p1p1p1p1p/c8/4P1PNP/P1P5C/7r1/9/1NBAKAB2 w"
fen2 = "2bakab1r/9/4R1n2/p1p1p1p1p/c8/4P1PNP/P1P5C/7r1/9/1NBAKAB2 b"
# _fen="r1bakabr1/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C1N2/9/RNBAKABR1 w - - 0 1"
# 解析FEN字符串并获取棋盘布局
initial_board = parse_fen(fen1)
final_board = parse_fen(fen2)
# 找出移动的起始点和终止点
move = find_move(initial_board, final_board)
print(move)
# 将起始点和终止点转换为象棋的移动表示形式
algebraic_notation = convert_to_algebraic_notation(move)

print("移动了", algebraic_notation)


def match(move):
    pattern = r'^[a-z]\d[a-z]\d$'
    if not re.match(pattern, move):
        print("不合法")
    else:
        print("合法")
match(algebraic_notation)
# 定义每个轴上的间隔数
num_points_x = 8
num_points_y = 2
num_points_z = 2

chess_eat_position = []
# 棋子的厚度
delta_z = -10
delta_x = weight_x = 20
delta_y = 20

for z in range(num_points_z):
    for y in range(num_points_y):
        for x in range(num_points_x):
            point_index = z * num_points_y * num_points_x + y * num_points_x + x
            coordinate = [(x + 1) * delta_x, y * delta_y, z * delta_z]
            chess_eat_position.append(coordinate)
print(chess_eat_position)

