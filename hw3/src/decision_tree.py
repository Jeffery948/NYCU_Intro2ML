"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
        self.feature_importance = [0] * X.shape[1]

    def _grow_tree(self, X, y, depth=0):
        cur_class = len(np.unique(y))
        if depth >= self.max_depth or cur_class == 1:
            check = sum(y)
            if check > len(y) / 2:
                return 1
            return 0
        node = find_best_split(X, y)
        left_data = node["left"]
        right_data = node["right"]
        left_child = self._grow_tree(left_data[:, 0:13], left_data[:, -1], depth + 1)
        right_child = self._grow_tree(right_data[:, 0:13], right_data[:, -1], depth + 1)
        node["left"] = left_child
        node["right"] = right_child
        return node

    def predict(self, X):
        pred = [self._predict_tree(x, self.tree) for x in X]
        self.compute_feature_importance(self.tree)
        return np.array(pred)

    def _predict_tree(self, x, tree_node):
        if isinstance(tree_node, dict):
            if x[tree_node["feature"]] <= tree_node["threshold"]:
                return self._predict_tree(x, tree_node["left"])
            return self._predict_tree(x, tree_node["right"])
        return tree_node

    def compute_feature_importance(self, node):
        if isinstance(node, dict):
            self.feature_importance[node["feature"]] += 1
            self.compute_feature_importance(node["left"])
            self.compute_feature_importance(node["right"])


# Split dataset based on a feature and threshold
def split_dataset(compose_data, feature_index, threshold):
    left, right = [], []
    for data in compose_data:
        if data[feature_index] <= threshold:
            left.append(data)
        else:
            right.append(data)
    return left, right


# Find the best split for the dataset
def find_best_split(X, y):
    feature_index, threshold, impurity, best_left, best_right = None, None, np.inf, None, None
    compose_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for index in range(len(thresholds) - 1):
            left, right = split_dataset(compose_data, feature, thresholds[index])
            left_y = [data[-1] for data in left]
            right_y = [data[-1] for data in right]
            total_len = len(left_y) + len(right_y)
            cur_impurity = (entropy(left_y) * len(left_y) + entropy(right_y) * len(right_y)) / total_len
            if cur_impurity < impurity:
                feature_index, threshold, impurity = feature, thresholds[index], cur_impurity
                best_left, best_right = np.array(left), np.array(right)
    return {"feature": feature_index, "threshold": threshold, "left": best_left, "right": best_right}


def entropy(y):
    y = np.array(y)
    class_1 = np.sum((y == 1))
    p = class_1 / len(y)
    if p == 1 or p == 0:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def gini_index(y):
    y = np.array(y)
    class_1 = np.sum((y == 1))
    p = class_1 / len(y)
    return 1 - p ** 2 - (1 - p) ** 2
