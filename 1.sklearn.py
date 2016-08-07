
from sklearn import tree
Features = [[140, 1], [130, 1], [150, 0], [170, 0]]
Labels = [1,1,0,0]
clf =tree.DecisionTreeClassifier()
clf = clf.fit(Features, Labels)
print clf.predict([150, 0])