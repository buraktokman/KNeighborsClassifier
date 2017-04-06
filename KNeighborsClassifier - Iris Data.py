#!/usr/bin/env python
'''
Pipeline
Train dataset for train learning model.
Test dataset to see how accurate it is on new data.
'''
#1 - Import a dataset
from sklearn import datasets
iris = datasets.load_iris()
'''
Think classifier as a function f(x) = y
f(x) -> Features / y -> Label
'''
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = .5)
# .5 means if we have 150 examples in Iris, 75 will be Train, 75 will be Test.

'''
#DecisionTreeClassifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
'''
#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
