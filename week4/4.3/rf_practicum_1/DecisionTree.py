import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode


class DecisionTree(object):
    '''
    A decision tree class.
    '''

    def __init__(self, impurity_criterion='entropy'):
        '''
        Initialize an empty DecisionTree.
        '''

        self.root = None  # root Node
        self.feature_names = None  # string names of features (for interpreting
                                   # the tree)
        self.categorical = None  # Boolean array of whether variable is
                                 # categorical (or continuous)
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini

    def fit(self, X, y, feature_names=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None
        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding
        label.
        feature_names is an optional list containing the names of each of the
        features.
        '''


        # This piece of code is used to provide feature names to the Decision tree
        if feature_names is None or len(feature_names) != X.shape[1]:
            # if the user has not provided feature names, just give them numbers
            self.feature_names = np.arange(X.shape[1])
        else:
            # otherwise, these are the names
            self.feature_names = feature_names

        # * Create True/False array of whether the variable is categorical
        # use a lambda function called is_categorical to determine if the variable is an instance
        # of str, bool or unicode - in that case is_categorical will be true
        # otherwise False. Look up the function isinstance()

        #is_categorical = lambda x: ?

        # Each variable (organized by index) is given a label categorical or not
        self.categorical = np.vectorize(is_categorical)(X[0])

        # Call the build_tree function
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - TreeNode
        Recursively build the decision tree. Return the root node.
        '''

        #  * initialize a root TreeNode

        # * set index, value, splits as the output of self._choose_split_index(X,y)

        # if no index is returned from the split index or we cannot split
        if index is None or len(np.unique(y)) == 1:
            # * set the node to be a leaf

            # * set the classes attribute to the number of classes
            # * we have in this leaf with Counter()

            # * set the name of the node to be the most common class in it

        else: # otherwise we can split (again this comes out of choose_split_index
            # * set X1, y1, X2, y2 to be the splits

            # * the node column should be set to the index coming from split_index

            # * the node name is the feature name as determined by
            #   the index (column name)

            # * set the node value to be the value of the split

            # * set the categorical flag of the node to be the category of the column

            # * now continue recursing down both branches of the split

        return node

    def _entropy(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float
        Return the entropy of the array y.
        '''

        total = 0
        # * for each unique class C in y
            # * count up the number of times the class C appears and divide by
            # * the total length of y. This is the p(C)
            # * add the entropy p(C) ln p(C) to the total
        return -total

    def _gini(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float
        Return the gini impurity of the array y.
        '''

        total = 0
        # * for each unique class C in y
            # * count up the number of times the class C appears and divide by
            # * the size of y. This is the p(C)
            # * add p(C)**2 to the total
        return 1 - total

    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)
        Return the two subsets of the dataset achieved by the given feature and
        value to split on.
        Call the method like this:
        X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)
        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        '''

        # * slice the split column from X with the split_index
        # * if the variable of this column is categorical
            # * select the indices of the rows in the column
            #  with the split_value (T/F) into one set of indices (call them A)
            # * select the indices of the rows in the column
            # that don't have the split_value into another
            #  set of indices (call them B)
        # * else if the variable is not categorical
             # * select the indices of the rows in the column
            #  less than the split value into one set of indices (call them A)
            # * select the indices of the rows in the column
            #  greater or equal to  the split value into
            # another set of indices (call them B)
        return X[A], y[A], X[B], y[B]

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float
        Return the information gain of making the given split.
        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        '''
        # * set total equal to the impurity_criterion

        # * for each of the possible splits y1 and y2
            # * calculate the impurity_criterion of the split
            # * subtract this value from the total, multiplied by split_size/y_size
        return total

    def _choose_split_index(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)
        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.
        Return None, None, None if there is no split which improves information
        gain.
        Call the method like this:
        index, value, splits = self._choose_split_index(X, y)
        X1, y1, X2, y2 = splits
        '''

        # set these initial variables to None
        split_index, split_value, splits = None, None, None
        # we need to keep track of the maximum entropic gain
        max_gain = 0

        # * for each column in X
            # * set an array called values to be the
            # unique values in that column (use np.unique)

            # if there are less than 2 values, move on to the next column
            if len(values) < 2:
                continue

            # * for each value V in the values array

                # * make a temporary split (using the column index and V) with make_split

                # * calculate the information gain between the original y, y1 and y2

                # * if this gain is greater than the max_gain
                    # * set max_gain, split_index, and split_value to be equal
                    # to the current max_gain, column and value

                    # * set the output splits to the current split setup (X1, y1, X2, y2)
        return split_index, split_value, splits

    def predict(self, X):
        '''
        INPUT:
            - X: 2d numpy array
        OUTPUT:
            - y: 1d numpy array
        Return an array of predictions for the feature matrix X.
        '''

        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        '''
        Return string representation of the Decision Tree. This will allow you to $:print tree
        '''
        return str(self.root)