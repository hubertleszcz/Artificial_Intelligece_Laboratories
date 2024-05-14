import copy

import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        if len(y) == 0:
            return best_idx, best_gain

        giniGain = 0
        giniLeft = 0
        giniRight = 0

        for i in possible_splits:
            leftPos = np.sum(y[:i-1] == 1)
            rightPos = np.sum(y[i+1:] == 1)
            leftNeg = np.sum(y[:i-1] == 0)
            rightNeg = np.sum(y[i+1:] == 0)

            # errors with div by 0
            if leftPos + leftNeg == 0:
                continue

            giniLeft = 1 - pow(leftPos / (leftNeg + leftPos), 2) - pow( leftNeg/(leftNeg + leftPos), 2)
            giniRight = 1 - pow(rightPos / (rightNeg + rightPos), 2) - pow( rightNeg/(rightNeg + rightPos), 2)

            left = leftPos + leftNeg
            right = rightNeg + rightPos

            giniGain = 1 - left * giniLeft / (left + right) - right*giniRight/(left+right)

            if giniGain > best_gain:
                best_gain = giniGain
                best_idx = i

        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None


        #TODO implement feature selection



        if feature_subset is None:
            iteration_range = range(X.shape[1])
        else:
            iteration_range = np.random.choice(X.shape[1], size=feature_subset, replace=False)

        for d in iteration_range:
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
