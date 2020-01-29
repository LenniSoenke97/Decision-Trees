import classification as c
import dataset as d

[x, y] = d.read_dataset('data/toy.txt')
dtc = c.DecisionTreeClassifier()
dtc.train(x, y)
print(dtc.plot())
