import numpy as np
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import accuracy_score

#dataset: x = [height in cm, weight in kg, EU shoe size], y = ['gender']
x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], 
	 [166,65,40], [190,90,47], [175,64,39], [177,70,40], 
	 [159,55,37], [171,75,42], [181,85,43]]
y = ['male', 'male', 'female', 'female', 'male', 'male',
	 'female', 'female', 'female', 'male', 'male']

#the following code initiates classifiers, trains them on the dataset, tests using the same data,
#and returns the accuracy score for each classification model

#decision tree classifier
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(x,y)
clf_tree_results = clf_tree.predict(x)
print('Decision tree accuracy: {}' .format(accuracy_score(y, clf_tree_results)*100))

#k-nearest neighbors classifier; n_neighbors=3 yields better accuracy than the default n_neighbors=5
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=3)
clf_knn = clf_knn.fit(x,y)
clf_knn_results = clf_knn.predict(x)
print('K-nearest neighbors accuracy: {}' .format(accuracy_score(y, clf_knn_results)*100))

#SVC classifier
clf_svc = svm.SVC()
clf_svc = clf_svc.fit(x,y)
clf_svc_results = clf_svc.predict(x)
print('SVC accuracy: {}' .format(accuracy_score(y, clf_svc_results)*100))

#AdaBoost classifier
clf_ada = ensemble.AdaBoostClassifier()
clf_ada = clf_ada.fit(x,y)
clf_ada_results = clf_ada.predict(x)
print('AdaBoost accuracy: {}' .format(accuracy_score(y, clf_ada_results)*100))