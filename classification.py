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
    A decision tree classifier
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False

    def find_best_node(self, feature_arr, label_arr, number_splits=1):

        assert (number_splits == 1)  # Only binary split implemented so far

        best_feature = 'n.a.'
        max_gain = -1
        thresholds = []

        # Calculate parent entropy
        parent_entropy = ep.parent_entropy([feature_arr, label_arr])

        # Iterate over features (rows of transposed feature_set)
        for feature in range(feature_arr.shape[1]):
            # Iterate over possible thresholds
            for threshold in np.unique(feature_arr[:, feature]):
                children_subsets = self.split_dataset([feature_arr, label_arr], feature, [threshold])
                info_gain = parent_entropy - ep.child_entropy(children_subsets, feature_arr.shape[0])
                if info_gain > max_gain:
                    best_feature = feature  # TODO Consider moving out of inner for loop
                    max_gain = info_gain
                    thresholds = [threshold]

        assert (max_gain >= 0)  # Assert best_feature and threshold were changed
        return best_feature, thresholds

    # Thresholds is array of threshold for which all less than or equal to threshold value will be put in set
    # "left" of split
    def split_dataset(self, parent_set, feature, thresholds):
        feature_arr = parent_set[0]
        label_arr = parent_set[1]
        assert feature_arr.shape[0] == label_arr.shape[0]

        # Prepare empty nested arrays for children subsets
        children_subsets = [[[], []] for _ in range(len(thresholds) + 1)]

        # Copy parent dataset rows into appropriate children datasets
        thresholds.sort()
        for row in range(len(feature_arr)):
            copied = False
            for split in range(len(thresholds)):
                if feature_arr[row][feature] <= thresholds[split]:
                    children_subsets[split][0].append(feature_arr[row])
                    children_subsets[split][1].append(label_arr[row])
                    copied = True
                    break
            if not copied:  # Value of node is not below any threshold for the row, append to last children subset
                children_subsets[-1][0].append(feature_arr[row])
                children_subsets[-1][1].append(label_arr[row])

        # Convert feature and label arrays to Numpy array type
        for split in range(len(thresholds) + 1):
            children_subsets[split][0] = np.asarray(children_subsets[split][0], parent_set[0].dtype)
            children_subsets[split][1] = np.asarray(children_subsets[split][1], parent_set[1].dtype)

        return children_subsets

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

        # Own

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

        # remember to change this if you rename the variable
        return predictions
