import gender_classifier
import numpy as np

#pass an array of your height in cm, weight in kg, and EU shoe size, e.g.: [181,80,44]
#the function will then classify your gender according to four different models
def test_your_measurements(your_array):
	x = np.array(your_array)
	x = x.reshape(1,-1)
	print('Decision tree classification: {}' .format(gender_classifier.clf_tree.predict(x)))
	print('K-nearest neighbors classification: {}' .format(gender_classifier.clf_knn.predict(x)))
	print('SVC classification: {}' .format(gender_classifier.clf_svc.predict(x)))
	print('AdaBoost classification: {}' .format(gender_classifier.clf_ada.predict(x)))

#I had way too much fun playing with this, so knock yourself out