import numpy as np
from classification import DecisionTreeClassifier
from eval import Evaluator


def k_split(dataset, k):
    """ splits a dataset into k folds
    
    Parameters
    ----------
    dataset : numpy.array of features and labels
    k  : int of number of folds wanted 
    
    Returns
    -------
        np array of split folds
    
        """


    index = np.random.permutation(dataset[0].shape[0])
    fold_index= np.split(index, k)

    return fold_index

    
def cross_validate(dataset, k):
    """ performs k fold cross validation on a dataset
    
    Parameters
    ----------
    dataset : numpy.array of features and labels
    k  : int of number of folds wanted 
    
    Returns
    -------
        an array of sorted tuples (tree, accuracy)
        sorted by accuracy(float)
        with tree being A DecisionTreeClassifier instance

    
        """
    
    treeAcc=[]
    fold_index = k_split(dataset, k)

    for valid_index in fold_index:

        evlu = Evaluator()
        tree = DecisionTreeClassifier()
        
        train_index = np.delete(np.arange(dataset[0].shape[0]),valid_index)
        
        valid_set= (dataset[0][valid_index,:],dataset[1][valid_index])
        train_set = (dataset[0][train_index,:],dataset[1][train_index])

        tree.train(train_set[0], train_set[1])
        predictions = tree.predict(valid_set[0])

        confusion = evlu.confusion_matrix(predictions, valid_set[1])

        treeAcc.append((tree, evlu.accuracy(confusion)))

    treeAcc= sorted(treeAcc, key=lambda Acc: -Acc[1])   # sort by accuracy 
    return treeAcc












    
