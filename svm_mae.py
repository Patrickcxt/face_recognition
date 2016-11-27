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
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
from collections import Counter

def train_age():
    features = pickle.load(open('./features/features_5w_precise'))
    labels = pickle.load(open('./features/age_labels_5w_precise'))
#labels = np.ones((100))

    features = features[0:20000]
    labels = labels[0:20000]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=2)

#path = './model/inceptionV3_100000.pkl'
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

    print("MAE:{0:0.3f}".format(mean_absolute_error(y_test, y_pred)))

    joblib.dump(clf, './model/inceptionV3_2w_precise.pkl')


def train_age2():
    """
    features = pickle.load(open('./wiki_opencv_features/features_precise'))
    age_labels = pickle.load(open('./wiki_opencv_features/age_labels_precise'))
    """

    features = pickle.load(open('./wiki_opencv_features/features_vgg_untuned'))
    age_labels = pickle.load(open('./wiki_opencv_features/age_labels_vgg_untuned'))

    features_2w = features[0:22000]
    age_labels_2w = age_labels[0:22000]

    features_3w = features[22000:]
    age_labels_3w = age_labels[22000:]

    print(len(features_3w))
    print(len(age_labels_3w))

    num_svm = 20
    clfs = []
    for i in range(num_svm):
        """
        clf = joblib.load('./age_model/wiki-opencv_2w_precise_' + str(i) + '.pkl')
        """
        clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr', tol=1e-6)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_2w, age_labels_2w, test_size=0.3, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("MAE:{0:0.3f}".format(mean_absolute_error(y_test, y_pred)))
        joblib.dump(clf, './age_model/wiki-opencv_vgguntuned_' + str(i) + '.pkl')
        clfs.append(clf)

    preds = np.zeros(len(features_3w))
    for i in range(num_svm):
        pred = clfs[i].predict(features_3w)
        preds = preds + pred

    preds = (preds // num_svm)
    print(preds)

    print("Final Accuracy:{0:0.3f}".format(mean_absolute_error(age_labels_3w, preds)))

    #joblib.dump(clf, './sex_model/inceptionV3_2w_precise_3.pkl')


def train_sex():
    features_1w = pickle.load(open('./imdb_features/features_1w_precise'))
    features_2w = pickle.load(open('./imdb_features/features_2w_precise'))
    features = np.concatenate((features_1w, features_2w), axis=0)
    #age_labels = pickle.load(open('./imdb_features/age_labels_1k_precise'))
    sex_labels_1w = pickle.load(open('./imdb_features/sex_labels_1w_precise'))
    sex_labels_2w = pickle.load(open('./imdb_features/sex_labels_2w_precise'))
    sex_labels = np.concatenate((sex_labels_1w, sex_labels_2w), axis=0)

    features_3w = pickle.load(open('./imdb_features/features_3w_precise'))
    sex_labels_3w = pickle.load(open('./imdb_features/sex_labels_3w_precise'))

    print(len(features_3w))
    #print(len(age_labels))
    print(len(sex_labels_3w))


    #path = './sex_model/inceptionV3_2w_precise.pkl'
    # path = ''
    """
    if path != '':
        clf = joblib.load(path)
    else:
        clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr', tol=1e-6)
        """

    clfs = []
    for i in range(19):
        clf = joblib.load('./sex_model/inceptionV3_2w_precise_' + str(i) + '.pkl')
        """
        clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr', tol=1e-6)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, sex_labels, test_size=0.3, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:{0:0.3f}".format(accuracy_score(y_test, y_pred)))
        joblib.dump(clf, './sex_model/inceptionV3_2w_precise_' + str(i) + '.pkl')
        """
        clfs.append(clf)

    preds = np.zeros(len(features_3w))
    for i in range(19):
        pred = clfs[i].predict(features_3w)
        preds = preds + pred
    preds = preds // 10

    print("Final Accuracy:{0:0.3f}".format(accuracy_score(sex_labels_3w, preds)))

    #joblib.dump(clf, './sex_model/inceptionV3_2w_precise_3.pkl')



def train_sex2():
    #features = pickle.load(open('./wiki_opencv_features/features_precise'))
    #sex_labels = pickle.load(open('./wiki_opencv_features/sex_labels_precise'))

    features = pickle.load(open('./wiki_opencv_features/features_vgg_untuned'))
    sex_labels = pickle.load(open('./wiki_opencv_features/sex_labels_vgg_untuned'))

    features_2w = features[0:22000]
    sex_labels_2w = sex_labels[0:22000]

    features_3w = features[22000:]
    sex_labels_3w = sex_labels[22000:]



    print(len(features_3w))
    print(len(sex_labels_3w))


    clfs = []
    svm_num = 9
    for i in range(svm_num):
        """

        clf = joblib.load('./sex_model/wiki-opencv_2w_precise_' + str(i) + '.pkl')
        """
        clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr', tol=1e-6)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_2w, sex_labels_2w, test_size=0.3, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:{0:0.3f}".format(accuracy_score(y_test, y_pred)))
        joblib.dump(clf, './sex_model/wiki-opencv_2w_precise_' + str(i) + '.pkl')
        clfs.append(clf)

    preds = np.zeros(len(features_3w))
    for i in range(svm_num):
        pred = clfs[i].predict(features_3w)
        preds = preds + pred
    preds = preds // (svm_num/2 + 1)

    print("Final Accuracy:{0:0.3f}".format(accuracy_score(sex_labels_3w, preds)))

    #joblib.dump(clf, './sex_model/inceptionV3_2w_precise_3.pkl')


train_sex2()
#train_sex()
