import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, auc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math
from treelib import Node, Tree


class DecisionTree:
    class Split:
        def __init__(self, col=None, value=None, gini_idx=None,
                     left_gini=None, right_gini=None, left_mode=None, right_mode=None) -> None:
            self.col = col # column index
            self.value = value
            self.gini_idx = gini_idx
            self.left_gini = left_gini
            self.right_gini = right_gini
            self.left_mode = left_mode
            self.right_mode = right_mode

        def __eq__(self, other) -> bool:
            return self.col == other.col and self.value == other.value and self.gini_idx == other.gini_idx

        def __lt__(self, other):
            return self.gini_idx < other.gini_idx

    class Node:
        def __init__(self, gini=None, clas=None) -> None:
            self.split = None
            self.left = None
            self.right = None
            self.clas = clas
            self.gini = gini

        def construct_tree(self, X, y, depth):
            if depth == 0 or math.isclose(self.gini, 0):
                return
            try:
                self.split = DecisionTree.find_best_split(X, y) 
            except DecisionTree.SplitNotPossibleException:
                return

            if depth is not None:
                depth -= 1

            self.left = DecisionTree.Node(self.split.left_gini, self.split.left_mode)
            leq_indices = X[:, self.split.col] <= self.split.value
            self.left.construct_tree(X[leq_indices], y[leq_indices], depth)

            self.right = DecisionTree.Node(self.split.right_gini, self.split.right_mode)
            gt_indices = X[:, self.split.col] > self.split.value
            self.right.construct_tree(X[gt_indices], y[gt_indices], depth)

        def predict(self, X):
            if self.split is None:
                return [self.clas] * len(X)
            leq_indices = X[:, self.split.col] <= self.split.value
            gt_indices = ~leq_indices
            y = np.empty(len(X))
            y[leq_indices] = self.left.predict(X[leq_indices])
            y[gt_indices] = self.right.predict(X[gt_indices])
            return y

        def get_text_tree(self, tree, parent) -> str:
            if self.split:
                node_text = f'X[{self.split.col}] <= {self.split.value:.2f}, gini_split={self.split.gini_idx:.2f}, gini={self.gini:.2f}' 
                tree.create_node(node_text, self, parent=parent)
                if self.left:
                    self.left.get_text_tree(tree, self)
                if self.right:
                    self.right.get_text_tree(tree, self)
            else:
                tree.create_node(f'class={self.clas}, gini={self.gini:.2f}', parent=parent)

    class SplitNotPossibleException(Exception):
        pass



    def __init__(self, max_depth=None) -> None:
        # branches by value (left less than or equal to, right larger than)
        self.root = None
        if max_depth is not None and max_depth < 0:
            raise Exception("depth must be >= 0")    
        self.max_depth = max_depth

    def fit(self, X, y):
        gini, mode = DecisionTree.calc_gini_of_set(y, return_mode=True)
        self.root = DecisionTree.Node(gini, mode)
        self.root.construct_tree(X, y, self.max_depth)

    def predict(self, X) -> list:
        return self.root.predict(X)

    def get_text_tree(self) -> str:
         tree = Tree()
         self.root.get_text_tree(tree, None)
         return tree

    # the training data are not necessary during prediction, so I am using static methods

    @staticmethod
    def find_best_split(X, y):
        best_split = DecisionTree.Split(gini_idx=1.1)
        for i in range(X.shape[1]):
            split = DecisionTree.find_best_split_for_col(X, y, i)
            best_split = min(best_split, split)

        if best_split.gini_idx > 1:
            # each column has only one unique value so there is no way to split the data
            raise DecisionTree.SplitNotPossibleException()
        return best_split

    # calc for all middles between col values
    @staticmethod
    def find_best_split_for_col(X, y, col_i):
        best_split = DecisionTree.Split(gini_idx=1.1)
        # construct unique sorted values of the column
        sorted_uq_vals = np.unique(X[:, col_i])
        for i in range(len(sorted_uq_vals) - 1):
            value = ((sorted_uq_vals[i] + sorted_uq_vals[i+1]) / 2)
            split = DecisionTree.calc_gini_index(X, y, col_i, value)
            best_split = min(best_split, split)
        return best_split

    @staticmethod
    def calc_gini_index(X, y, col_i, value):
        leq_indices = X[:, col_i] <= value
        gt_indices = ~leq_indices
        left_y = y[leq_indices]
        right_y = y[gt_indices]
        left_gini, left_mode = DecisionTree.calc_gini_of_set(left_y, return_mode=True)
        right_gini, right_mode = DecisionTree.calc_gini_of_set(right_y, return_mode=True)
        gini_idx = len(left_y) / len(y) * left_gini + \
            len(right_y) / len(y) * right_gini
        return DecisionTree.Split(col_i, value, gini_idx, left_gini, right_gini, left_mode, right_mode)

    @staticmethod
    def calc_gini_of_set(y, return_mode=False):
        uniques, counts = np.unique(y, return_counts=True)
        gini = 1 - np.sum((counts / len(y))**2)
        if return_mode:
            mode = uniques[np.argmax(counts)]
            return gini, mode
        return gini    



if __name__ == "__main__":
    X = np.genfromtxt('data_classification/iris.csv', delimiter=';')
    X, y = X[:, :-1], X[:, -1]


    clf = DecisionTree()
    clf.fit(X, y)
    clf.get_text_tree().show()

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    #plt.show()

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=13)
    clf = DecisionTree()
    clf.fit(X_train, y_train)
    clf.get_text_tree().show()
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc2 = sum(1 if x else 0 for x in y_test == y_pred) / len(y_test)
    print(f'acc={acc}, acc2={acc2}')

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc2 = sum(1 if x else 0 for x in y_test == y_pred) / len(y_test)
    print(f'acc={acc}, acc2={acc2}')


    df = pd.read_csv('data_classification/titanic_preprocessed.csv', index_col=0)
    X, y = df.loc[:, df.columns != 'Survived'], df.loc[:, 'Survived']
    X, y = X.values, y.values

    def kfold_classification(clf, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits)
        scores = []
        train_scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))

            y_train_pred = clf.predict(X_train)
            train_scores.append(accuracy_score(y_train, y_train_pred))
        return np.mean(scores), np.mean(train_scores)

    

    scores = []
    for i in range(1, 16):
        clf = DecisionTree(max_depth=i)
        score, train_score = kfold_classification(clf)
        scores.append((i, score, train_score))
    plt.figure()
    df_scores = pd.DataFrame(scores, columns=['depth', 'score', 'train_score'])
    sns.lineplot(data=df_scores, x='depth', y='score', legend='brief', label='score')
    sns.lineplot(data=df_scores, x='depth', y='train_score', legend='brief', label='train_score')
    plt.show()



    pass
