fen = "r1bakabr1/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C1N2/9/RNBAKAB1R w - - 0 1"

def count_total_pieces(fen):
    total_pieces = 0
    fen_parts = fen.split(' ')[0]
    for row in fen_parts.split('/'):
        for char in row:
            if char.isalpha():
                total_pieces += 1
    return total_pieces

total_pieces = count_total_pieces(fen)
print("所有棋子数量：", total_pieces)