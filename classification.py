##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np
import entropy as ep


class DecisionTreeClassifier(object):
    """
    A binary decision tree classifier for integer values
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    prediction : int
        Predicted label of classifier (None for all but leaf nodes)
    feature : int
        Feature array index of node feature
    threshold :
        Node threshold (<threshold classified by left child, >= classified by right child)
    left_child :
        Left child classifier (None for leaf nodes)
    right_child :
        Right child classifier (None for leaf nodes)
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    plot()
        Plots decision tree
    """

    def __init__(self):
        self.is_trained = False
        self.prediction = None
        self.feature = None
        self.threshold = None
        self.left_child = None
        self.right_child = None

    def train(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the 
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array
        
        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        [best_feature, best_threshold] = find_best_node(x, y)
        if best_feature is None:
            (label, count) = np.unique(y, return_counts=True)
            self.prediction = label[np.argmax(count)]
        else:
            self.feature = best_feature
            self.threshold = best_threshold
            children_subsets = split_dataset([x, y], best_feature, best_threshold)
            self.left_child = DecisionTreeClassifier()
            self.right_child = DecisionTreeClassifier()
            self.left_child.train(children_subsets[0][0], children_subsets[0][1])
            self.right_child.train(children_subsets[1][0], children_subsets[1][1])

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        
        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # If prediction is set (regardless of data), return prediction
        if self.prediction is not None:
            return np.full((x.shape[0],), fill_value=self.prediction, dtype=np.object)

        for sample_idx in range(x.shape[0]):
            if x[sample_idx, self.feature] < self.threshold:
                predictions[sample_idx] = self.left_child.predict(np.array([x[sample_idx]]))[0]
            else:
                predictions[sample_idx] = self.right_child.predict(np.array([x[sample_idx]]))[0]

        # remember to change this if you rename the variable
        return predictions

    def plot(self, level=0):
        string = ''
        indent = '   ' * level
        if level > 0:
            string += indent
        string += '+--'
        if self.prediction is None:
            string += ('x[' + str(self.feature) + '] < ' + str(self.threshold) + ':\n'
                       + self.left_child.plot(level + 1) + '\n' + indent + '+--x[' + str(self.feature) + '] >= '
                       + str(self.threshold) + ':\n' + self.right_child.plot(level + 1))
        else:
            string += ' ' + str(self.prediction)
        return string


def split_dataset(parent_set, feature, threshold):
    """
        Splits input parent_set dataset into two children subsets according to feature and threshold
    """

    num_children_subsets = 2
    feature_arr = parent_set[0]
    label_arr = parent_set[1]
    assert feature_arr.shape[0] == label_arr.shape[0]

    # Prepare empty nested arrays for children subsets
    children_subsets = [[[], []] for _ in range(num_children_subsets)]

    # Copy parent dataset rows into appropriate children dataset
    for row in range(len(feature_arr)):
        if feature_arr[row][feature] < threshold:
            children_subsets[0][0].append(feature_arr[row])
            children_subsets[0][1].append(label_arr[row])
            copied = True
        else:
            children_subsets[1][0].append(feature_arr[row])
            children_subsets[1][1].append(label_arr[row])

    # Convert feature and label arrays to Numpy array type
    for child in range(num_children_subsets):
        children_subsets[child][0] = np.asarray(children_subsets[child][0], parent_set[0].dtype)
        children_subsets[child][1] = np.asarray(children_subsets[child][1], parent_set[1].dtype)

    return children_subsets


def find_best_node(feature_arr, label_arr):
    """
        Finds best node (highest information gain) of feature and label arrays
    """

    best_gain = 0
    best_feature = None
    best_threshold = None

    # Calculate parent entropy
    parent_entropy = ep.parent_entropy([feature_arr, label_arr])

    # Iterate over features (rows of transposed feature_set)
    for feature in range(feature_arr.shape[1]):
        # Iterate over possible thresholds
        for threshold in np.unique(feature_arr[:, feature]):
            children_subsets = split_dataset([feature_arr, label_arr], feature, threshold)
            info_gain = parent_entropy - ep.child_entropy(children_subsets, feature_arr.shape[0])
            if info_gain > best_gain:
                best_feature = feature
                best_gain = info_gain
                best_threshold = threshold

    return best_feature, best_threshold
