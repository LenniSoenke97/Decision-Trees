import numpy as np

def k_split(dataset, k):

    fold_size = dataset.shape[0]/k
    index = np.random.permutation(dataset.shape[0])
    fold_split = np.split(index, k)

    return fold_split

    
    
