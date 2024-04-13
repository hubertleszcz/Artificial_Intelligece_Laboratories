from exceptions import *
from connect4 import Connect4
import copy


class AlphaBetaAgent:

    def __init__(self, player):
        self.my_token = player
        if self.my_token == 'x':
            self.opponent_token = 'o'
        else:
            self.opponent_token = 'x'

    def heuristics(self, connect4):
        result = 0
        allCombos = connect4.iter_fours()
        for i in allCombos:
            myTokens = i.count(self.my_token)
            opponentTokens = i.count(self.opponent_token)
            blankSpaces = i.count('_')
            if myTokens == 4:
                result = result + 1
            elif opponentTokens == 4:
                result = result-1
            elif myTokens == 3 and blankSpaces == 1:
                result = result + 0.2
            elif opponentTokens == 3 and blankSpaces == 1:
                result = result - 0.2
            elif myTokens == 2 and blankSpaces == 2:
                result = result + 0.02
            elif opponentTokens == 2 and blankSpaces == 2:
                result = result - 0.02
            elif myTokens == 2 and opponentTokens == 1:
                result = result + 0.001
            elif opponentTokens and myTokens == 1:
                result = result - 0.001
        return result

    def alphaBeta(self, connect4, x, d, alpha, beta):
        if connect4.game_over:
            if connect4.getWins() == self.my_token:
                return 1
            elif connect4.getWins() is None:
                return 0
            else:
                return -1
        elif d == 0:
            return self.heuristics(connect4)
        else:
            if x == 1:
                v = -float('inf')
                for i in range(len(connect4.possible_drops())):
                    tmpConnect4 = copy.deepcopy(connect4)
                    try:
                        tmpConnect4.drop_token(connect4.possible_drops()[i])
                        value = self.alphaBeta(tmpConnect4, 0, d - 1, alpha, beta)
                        v = max(value, v)
                        alpha = max(alpha, v)
                        if v >= beta:
                            break
                    except GameplayException:
                        pass
                return v
            else:
                v = float('inf')
                for i in range(len(connect4.possible_drops())):
                    tmpConnect4 = copy.deepcopy(connect4)
                    try:
                        tmpConnect4.drop_token(connect4.possible_drops()[i])
                        value = self.alphaBeta(tmpConnect4, 1, d - 1, alpha, beta)
                        v = min(value, v)
                        beta = min(beta, v)
                        if v <= beta:
                            break
                    except GameplayException:
                        pass
                return v

    def decide(self, connect4):
        bestResult = -float('inf')
        bestCol = None
        possible = connect4.possible_drops()
        for col in possible:
            new_game = copy.deepcopy(connect4)
            try:
                new_game.drop_token(col)
                result = self.alphaBeta(new_game, 0, 5, -float('inf'), float('inf'))
                if result > bestResult:
                    bestResult = result
                    bestCol = col
            except GameplayException:
                continue
        return bestCol

