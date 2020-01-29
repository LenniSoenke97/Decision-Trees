import numpy as np
import crossVal
import dataset

data = dataset.read_dataset("data/train_full.txt")

treeAcc = crossVal.cross_validate(data, 10);

print(treeAcc)
