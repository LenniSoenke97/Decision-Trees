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

        def find_best_node(feature_set, label_set, number_splits=2):
            assert (number_splits == 2)  # Only binary split implemented so far
            best_node = 'n.a.'  # Need to sort edge cases
            max_gain = 0
            thresholds = []

            # parent_entropy = parent_entropy([feature_set, label_set]) ! This is calculated repeatedly by
            # information_gain()

            # Iterate over features (rows of transposed feature_set)
            for node in range(feature_set.shape[1]):
                # Iterate over possible thresholds
                for threshold in feature_set[:, node]:
                    children_subsets = split_dataset([feature_set, label_set], node, [threshold])
                    info_gain = ep.information_gain([feature_set, label_set], children_subsets)
                    if info_gain > max_gain:
                        best_node = node
                        max_gain = info_gain
                        thresholds = [threshold]

            # Need to sort edge case
            assert (best_node != 'n.a.')
            return best_node, thresholds

        # Thresholds is array of threshold (integer value)
        def split_dataset(parent_set, node, thresholds):
            return  # Array of children_subsets

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
