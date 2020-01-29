import numpy as np
import crossVal
import dataset
from classification import DecisionTreeClassifier
from eval import Evaluator

#Question 3.3

train_full = dataset.read_dataset("data/train_full.txt")
treeAcc = crossVal.cross_validate(train_full, 10);
acc = np.asarray(treeAcc[1])

print("Accuracy: {:.4f} +- {:.4f} ".format(np.mean(acc), np.std(acc)))

#Question 3.4

topTree = treeAcc[0][0]
topAcc = treeAcc[0][1]

test_set = dataset.read_dataset("data/test.txt")
fullTree = DecisionTreeClassifier()
evlu = Evaluator()

fullTree.train(train_full[0], train_full[0])
predictions = fullTree.predict(test_set[0])

fullConfusion = evlu.confusion_matrix(predictions, test_set[1])
fullAcc = evlu.accuracy(fullConfusion)

if (fullAcc > topAcc):
    tree_string = " tree trained on the full Data Set "
else:
    tree_string = " top tree in the k fold cross-validation "

print("""Top k fold cross validated Tree Accuracy: %.4f, 
         Tree trained on full set Accuracy: %.4f, 
         the %s has the highest Accuracy  
         on the test set.""" % (topAcc, fullAcc, tree_string))

#Question 3.5
trees = treeAcc[0]
running_preds=[]

for tree in trees:
    running_preds.append(tree.predict(test_set[0]))

predictions =  np.transpose(np.asarray(running_preds))
comb_preds=[]

for pred in predictions:
    comb_preds.append(max(pred, key = pred.count))

combConfusion = evlu.confusion_matrix(np.asarray(comb_preds), test_set[1])
combAcc = evlu.accuracy(combConfusion)

if (fullAcc > combAcc):
    tree_string = " tree trained on the full Data Set "
else:
    tree_string = " combined trees from the k fold cross-validation "

print("""Combine k fold cross validated Tree Accuracy: %.4f, 
         Tree trained on full set Accuracy: %.4f, 
         the %s has the highest Accuracy  
         on the test set.""" % (topAcc, combAcc, tree_string))



        

       


