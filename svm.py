import os
import re
import sys

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
from collections import Counter


def train_age():

    print("Trainging SVM for age...")

    features = pickle.load(open('./wiki_opencv_features/features_vgg_tuned_5000'))
    age_labels = pickle.load(open('./wiki_opencv_features/age_labels_vgg'))

    features_2w = features[0:22000]
    age_labels_2w = age_labels[0:22000]

    features_3w = features[22000:]
    age_labels_3w = age_labels[22000:]

    num_svm = 20
    clfs = []
    for i in range(num_svm):
        svm_path = './age_model/wiki-opencv_vggtuned_5k_' + str(i) + '.pkl'
        if os.path.exists(svm_path):
            clf = joblib.load(svm_path)
        else:
            clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr', tol=1e-4)
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_2w, age_labels_2w, test_size=0.3, random_state=i)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("MAE:{0:0.3f}".format(mean_absolute_error(y_test, y_pred)))
            joblib.dump(clf, svm_path)
            clfs.append(clf)

    preds = np.zeros(len(features_3w))
    for i in range(num_svm):
        pred = clfs[i].predict(features_3w)
        preds = preds + pred

    preds = (preds // num_svm)
    print("Final MAE:{0:0.3f}".format(mean_absolute_error(age_labels_3w, preds)))


def train_sex():

    print("Training SVM for sex...")

    # Load features
    features = pickle.load(open('./wiki_opencv_features/features_vgg_tuned_sex_1000'))
    sex_labels = pickle.load(open('./wiki_opencv_features/sex_labels_vgg'))

    features_2w = features[0:22000]
    sex_labels_2w = sex_labels[0:22000]

    features_3w = features[22000:]
    sex_labels_3w = sex_labels[22000:]


    # Train SVMs
    clfs = []
    svm_num = 19
    for i in range(svm_num):
        svm_path = './sex_model/wiki-opencv_1k_vggtuned_' + str(i) + '.pkl'
        if os.path.exists(svm_path):
            print("Loding " + svm_path)
            clf = joblib.load(svm_path)
        else:
            prnt("Training " + svm_path)
            clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr', tol=1e-6)
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_2w, sex_labels_2w, test_size=0.3, random_state=i)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("Accuracy:{0:0.3f}".format(accuracy_score(y_test, y_pred)))
            joblib.dump(clf, svm_path)
        clfs.append(clf)

    # Predict on the test set
    preds = np.zeros(len(features_3w))
    for i in range(svm_num):
        pred = clfs[i].predict(features_3w)
        preds = preds + pred
    preds = preds // (svm_num/2 + 1)

    print("Final Accuracy:{0:0.3f}".format(accuracy_score(sex_labels_3w, preds)))



if __name__ == '__main__':

    if sys.argv[1] == 'sex':
        train_sex()
    else:
        train_age()

