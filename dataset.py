##############################################################################
# CO395: Introduction to Machine Learning
# Dataset input reading function
# Prepared by: ms519@ic.ac.uk
##############################################################################

import numpy as np

########################### FUNCTION DEFINITIONS #############################

def read_dataset(filename):

    """
    Function reading in a dataset.

    Args:
        filename(string): name / location of the file

    Returns:
        feature_array(list[int]): array with the sample features
        label_array(list[char]): array with the sample lables
    """

    data_list = []
    with open(filename) as f:
        for line in f:
            data_list.append(line.split(","))

    conv_array = np.array(data_list)
    feature_array = conv_array[:,: - 1]
    label_array = conv_array[:, -1]

    count = 0
    
    for i in label_array:
        label_array[count] = i[0]
        count += 1
    
    return feature_array.astype(np.int8), label_array

'''
############################## FUNCTION TESTS ################################

def TEST_search(sample, noisy_feature):
    
    for i in range(0, (noisy_feature.size - 1)):
        if np.array_equal(sample, noisy_feature[i]):
            return i
    return -1

def TEST_different(full_feature, full_label, noisy_feature, noisy_label):
    count = 0
    false_list = []

    for sset in full_feature:
            line = TEST_search(sset, noisy_feature)
            if line != -1 and full_label[line] != noisy_label[line]:
                false_list.append(full_label[line])
                count += 1
    return count, false_list

full_feature, full_label = read_dataset("./data/train_full.txt")
sub_feature, sub_label = read_dataset("./data/train_sub.txt")
noisy_feature, noisy_label = read_dataset("./data/train_noisy.txt")

print("Full  Size: ", full_label.size)
print("Sub   Size: ", sub_label.size)
print("Noisy Size: ", noisy_label.size)

a = np.unique(full_feature, return_counts=True)
a_distribution = a[1] / full_label.size
b = np.unique(sub_feature, return_counts=True)
b_distribution = b[1] / sub_label.size
c = np.unique(full_label, return_counts=True)
c_distribution = c[1] / full_label.size
d = np.unique(sub_label, return_counts=True)
d_distribution = d[1] / sub_label.size

e = np.unique(noisy_feature, return_counts=True)
e_distribution = e[1] / noisy_label.size
f = np.unique(noisy_label, return_counts=True)
f_distribution = f[1] / noisy_label.size

print("Full   Attribute Values: ", np.around(a_distribution, 5)*100)
print("Sub    Attribute Values: ", np.around(b_distribution, 5)*100)
print("Noisy  Attribute Values: ", np.around(e_distribution, 5)*100)

print("Full  Label Values: ", np.around(c_distribution, 5)*100)
print("Sub   Label Values: ", np.around(d_distribution, 5)*100)
print("Noisy Label Values: ", np.around(f_distribution, 5)*100)

false_count, false_letters = TEST_different(full_feature, full_label, noisy_feature, noisy_label)
g = np.unique(np.asarray(false_letters), return_counts=True)
print("Noisy Percentage: ", false_count / full_label.size)
print((g[1]/np.sum(g[1]))*100)
'''


