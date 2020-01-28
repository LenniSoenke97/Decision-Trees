import numpy as np
from Classification import DecisionTreeClassifier
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


    fold_size = dataset.shape[0]/k
    index = np.random.permutation(dataset.shape[0])
    fold_split = np.split(index, k)

    return fold_split

    
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
    fold_split = k_split(dataset, k)

    for valid_idx in fold_idx:

        valid_set = dataset[valid_idx,:]
        train_set = dataset[np.delete(np.arange(dataset.shape[0]), valid_idx),:]

        tree = DecisionTreeClassifier()
        tree.train(train_set[:,:15], train_set[:,:-1])
        predictions = tree.predict(valid_set[:,:15])
        confusion = Evaluator.confusion_matrix(predictions, valid_set[:,-1])

        treeAcc.append((tree, Evaluator.accuracy(confusion)))

    treeAcc= sorted(treeAcc, key=lambda Acc: Acc[1])   # sort by accuracy 
    return treeAcc












    
