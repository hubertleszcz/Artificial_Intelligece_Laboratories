from exceptions import *
from connect4 import Connect4
import copy


class MinMaxAgent:

    def __init__(self, player, heuristicOn):
        self.my_token = player
        if self.my_token == 'x':
            self.opponent_token = 'o'
        else:
            self.opponent_token = 'x'
        self.heuristicOn = heuristicOn

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
                result = result - 1
            elif myTokens == 3 and blankSpaces == 1:
                result = result + 0.2
            elif opponentTokens == 3 and blankSpaces == 1:
                result = result - 0.2
            elif myTokens == 2 and blankSpaces == 2:
                result = result + 0.02
            elif opponentTokens == 2 and blankSpaces == 2:
                result = result - 0.02
        return result

    def M(self, connect4, x, d):
        if connect4.game_over:
            if connect4.getWins() == self.my_token:
                return 1
            elif connect4.getWins() is None:
                return 0
            else:
                return -1
        elif d == 0:
            if self.heuristicOn:
                return self.heuristics(connect4)
            return 0
        else:
            if x == 1:
                best_result = -2
                for i in connect4.possible_drops():
                    tmpConnect4 = copy.deepcopy(connect4)
                    try:
                        tmpConnect4.drop_token(i)
                        value = self.M(tmpConnect4, 0, d-1)
                        best_result = max(value, best_result)
                    except GameplayException:
                        pass
                return best_result
            else:
                best_result = 2
                for i in connect4.possible_drops():
                    tmpConnect4 = copy.deepcopy(connect4)
                    try:
                        tmpConnect4.drop_token(i)
                        value = self.M(tmpConnect4, 1, d-1)
                        best_result = min(value, best_result)
                    except GameplayException:
                        pass
                return best_result

    def decide(self, connect4):
        bestResult = -float('inf')
        bestCol = None
        possible = connect4.possible_drops()
        for col in possible:
            new_game = copy.deepcopy(connect4)
            try:
                new_game.drop_token(col)
                result = self.M(new_game, 0, 3)
                if result > bestResult:
                    bestResult = result
                    bestCol = col
            except GameplayException:
                continue
        return bestCol
