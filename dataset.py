##############################################################################
# CO395: Introduction to Machine Learning
# Dataset input reading function
# Prepared by: Max Strieth-Kalthoff
#
##############################################################################
import numpy as np

# Description: Function to read a given dataset
# Input:       filename
# Output:      numpy array of the features
#              numpy array of the labels

def read_dataset(filename):

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
    
    return feature_array.astype(int), label_array
