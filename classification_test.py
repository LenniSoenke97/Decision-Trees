import unittest
import classification as c
import dataset as d
import numpy as np


class TestClassification(unittest.TestCase):

    # Load data (toy dataset)
    [feature_arr, label_arr] = d.read_dataset('data/toy.txt')
    dtc = c.DecisionTreeClassifier()

    def test_split_dataset(self):
        feature = 1
        thresholds = [4]
        self.dtc.find_best_node(self.feature_arr, self.label_arr)
        actual = self.dtc.split_dataset([self.feature_arr, self.label_arr], feature, thresholds)
        expected = [[np.array([[1, 3, 1], [2, 1, 2], [5, 2, 6]], dtype=np.int8),
                     np.array(['C', 'C', 'C'])],
                    [np.array([[5, 7, 1], [4, 6, 2], [4, 6, 3], [1, 6, 3], [0, 5, 5], [1, 5, 0], [2, 4, 2]], dtype=np.int8),
                     np.array(['A', 'A', 'A', 'A', 'A', 'C', 'C'])]]
        # Assert integer values equal
        self.assertEqual(actual[0][0].all(), expected[0][0].all())
        self.assertEqual(actual[1][0].all(), expected[1][0].all())
        # Assert character values equal
        self.assertEqual(len(actual[0][1]), len(expected[0][1]))
        for idx in range(len(actual[0][1])):
            self.assertEqual(actual[0][1][idx], expected[0][1][idx])
        self.assertEqual(len(actual[1][1]), len(expected[1][1]))
        for idx in range(len(actual[1][1])):
            self.assertEqual(actual[1][1][idx], expected[1][1][idx])

    def test_find_best_node(self):
        # Split dataset
        [actual_feature, actual_threshold] = self.dtc.find_best_node(self.feature_arr, self.label_arr)
        expected = [1, [5]]
        self.assertEqual([actual_feature, [actual_threshold]], expected)
