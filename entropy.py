##############################################################################
# CO395: Introduction to Machine Learning 
# Entropy Calculation Module
# Prepared by: ms519@ic.ac.uk
##############################################################################

import numpy as np
import dataset as ds

########################### FUNCTION DEFINITIONS #############################

def parent_entropy(parent_set):

    """
    Function calculating parent entropy of a label set.

    Args:
        parent_set(list[np.array]): parent feature and label list

    Returns:
        p_entropy(int): information entropy of the label list
    """
    
    label_set = parent_set[1]
    total_count = label_set.size
    features = np.unique(label_set, return_counts=True)

    p_entropy = 0

    for i in features[1]:
        p_i = i/total_count
        p_entropy -= p_i * np.log2(p_i)
    
    return p_entropy

def child_entropy(children_subsets, total_rows):

    """
    Function calculating the total information entropy of given children subsets.

    Args:
        children_subsets(list[[np.array], [np.array]]): children feature and label list
        total_rows(int): row size of the current parent

    Returns:
        p_entropy(int): information entropy of the label list
    """

    c_entropy = 0
    
    for sset in children_subsets:
        c_entropy += (sset[1].size/total_rows) * parent_entropy(sset)

    return c_entropy

def information_gain(parent_set, children_subsets):
    
    """
    Function calculating the information gain for a given parent sets and its children subsets.

    Args:
        parent_set(list[np.array]): parent feature and label list
        children_subsets(list[[np.array, np.array]]): list of children feature and label lists

    Returns:
        ig_score(int): information gain of a given split
    """
    p = parent_entropy(parent_set)
    p_size = parent_set[1].size
    c = child_entropy(children_subsets, p_size)
    
    ig_score = p - c

    return ig_score

"""
############################## FUNCTION TESTS ################################

print("\n##############################")
print("Entropy module Function Tests:")
print("##############################\n")

# TEST: calculating the information entropy of the toy data set

print("Testing function parent_entropy(...)")
print("")

toy_set = []
toy_set_features, toy_set_labels = ds.read_dataset("./data/toy.txt")

toy_set.append(toy_set_features)
toy_set.append(toy_set_labels)
info_entropy = parent_entropy(toy_set)

print("Information Entropy of unsplitted 'toy.txt': ", info_entropy)
print("")

# TEST: calculating the information entropy of sample children subsets of toy data set

print("Testing function children_entropy(...)")
print("")

toy_set_features, toy_set_labels = ds.read_dataset("./data/toy.txt")

toy_subsets = []
child_subset_1 = [toy_set_features[0], np.array([toy_set_labels[0]])]
child_subset_2 = [toy_set_features[1:], toy_set_labels[1:]]
toy_subsets.append(child_subset_1)
toy_subsets.append(child_subset_2)

c_entropy = child_entropy(toy_subsets, toy_set_labels.size)

print("Information Entropy of splitted 'toy.txt' children: ", c_entropy)
print("")

# TEST: calculating the information gain of a parent set and its children subsets

print("Testing function information_gain(...)")
print("")

toy_set = []

toy_set_features, toy_set_labels = ds.read_dataset("./data/toy.txt")

toy_set.append(toy_set_features)
toy_set.append(toy_set_labels)

toy_subsets = []

child_subset_1 = [toy_set_features[0], np.array([toy_set_labels[0]])]
child_subset_2 = [toy_set_features[1:], toy_set_labels[1:]]
toy_subsets.append(child_subset_1)
toy_subsets.append(child_subset_2)

info_score = information_gain(toy_set, toy_subsets)

print("Information gain of current 'toy.txt split: ", info_score)
print("")

"""
