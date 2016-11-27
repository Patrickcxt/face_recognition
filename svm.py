import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib

features_10w = pickle.load(open('./features/features_100000'))
labels_10w = pickle.load(open('./features/age_labels_100000'))
features_15w = pickle.load(open('./features/features_10w_15w'))
labels_15w = pickle.load(open('./features/age_labels_10w_15w'))
features_20w = pickle.load(open('./features/features_15w_20w'))
labels_20w = pickle.load(open('./features/age_labels_15w_20w'))

features = np.concatenate((features_10w, features_15w), axis=0)
features = np.concatenate((features, features_20w), axis=0)
labels = np.concatenate((labels_10w, labels_15w), axis=0)
labels = np.concatenate((labels, labels_20w), axis=0)

#labels = np.ones((100))

print(len(features))
print(len(labels))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)

path = ''
if path != '':
    clf = joblib.load(path)
else:
    clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr', tol=1e-6)

clf.fit(X_train, y_train)
for i in range(len(X_test)):
    pred = clf.predict(X_test[i])
    print(y_test[i], pred[0])
y_pred = clf.predict(X_test)

print("Accuracy:{0:0.1f}%".format(accuracy_score(y_test, y_pred)*100))

joblib.dump(clf, './model/inceptionV3_20w.pkl')
