##############################################################################
# CO395: Introduction to Machine Learning
# Entropy calculation library
# Prepared by: Sander Coates, Max Strieth-Kalthoff
##############################################################################

import numpy as np
import dataset as ds

def parent_entropy(parent_set):
    
    feature_set = parent_set[0][1]
    total_count = np.size(feature_set, 0)
    features = np.unique(feature_set, return_counts=True)

    p_entropy = 0

    for i in features[1]:
        p_i = i/total_count
        p_entropy -= p_i * np.log2(p_i)
    
    return p_entropy

# Subsets is an array of feature / label sets
def child_entropy(children_subsets):
    return

def information_gain(parent_set, children_subsets):
    return

a=[]
f_a, l_a = ds.read_dataset("./data/toy.txt")
a.append([f_a, l_a])
x = parent_entropy(a)
print(x)