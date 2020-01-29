##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """
    
    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """
        
        if not class_labels:
            class_labels = np.unique(annotation)
        
        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        i=0
        for pred in prediction:
            prediction_class = list(class_labels).index(pred)
            annotation_class = list(class_labels).index(annotation[i])
            confusion [annotation_class][prediction_class] += 1
            '''
            if prediction_class == annotation_class:
                confusion[prediction_class][prediction_class] += 1
            else:
                confusion[prediction_class][annotation_class] += 1
                confusion[annotation_class][prediction_class] += 1
            '''

            i = i + 1
        
        
        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################
        
        
        return confusion
    
    
    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
        
        # feel free to remove this
        return confusion.trace()/confusion.sum()
            


        
        

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################
        
        return accuracy
        
    
    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """
        
        # Initialise array to store precision for C classes
        #p = np.zeros((len(confusion), 1))
        p = []


        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        i = 0
        for label in confusion:
            true_positive = confusion[i][i]
            rest = confusion.sum(axis=1)[i]
            p.append(true_positive/rest)
            i += 1

        # You will also need to change this        
        macro_p = np.average(p)

        return (p, macro_p)
    
    
    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        r = []
        i = 0
        for label in confusion:
            true_positive = confusion[i][i]
            rest = confusion.sum(axis=0)[i]
            r.append(true_positive/rest)
            i += 1
        
        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################
        
        # You will also need to change this        
        macro_r = np.average(r)
        
        return (r, macro_r)
    
    
    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        f = []
        precision, p = self.precision(confusion)
        recall, r = self.recall(confusion)
        for i in range(0,len(confusion)):
            f.append((2*precision[i]*recall[i])/(precision[i]+recall[i]))
            i += 1
        
        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################
        
        # You will also need to change this        
        macro_f = np.average(f)
        
        return (f, macro_f)
   
 
