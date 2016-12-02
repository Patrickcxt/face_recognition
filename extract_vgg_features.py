import os
import re
import sys

import tensorflow as tf
import tensorlayer as tl
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle

from scipy.misc import imread, imresize
from scipy.misc.pilutil import imshow

from vgg16 import vgg16


def get_train_set():
    print("Get training set ...")
    #images_dir = '/home/amax/cxt/data/IMDB_WIKI/imdb_face_opencv/'
    #images_dir = '/home/amax/cxt/data/IMDB_WIKI/imdb_data/'
    images_dir = '/home/amax/cxt/data/IMDB_WIKI/wiki_face_opencv/'
    list_images = [f for f in os.listdir(images_dir) if re.search('jpg', f)]
    images = []
    sex_labels = []
    age_labels = []

    for ind, image in enumerate(list_images):

        if image.find('NaN') != -1:
            continue

        label = image.split('.')
        label = label[0].split('_')
        age_label = int(label[2])
        sex_label = int(label[1])

        if age_label < 0 or age_label > 100:
            continue
        else:
            age_labels.append(age_label)
            sex_labels.append(sex_label)
        image = images_dir + image
        if not gfile.Exists(image):
            tf.logging.fatal("File does not exist %s", image)
            continue
        images.append(image)

    return images, sex_labels, age_labels




def main(mode, load_path, save_path):
    # Get dataset
    images, sex_labels, age_labels = get_train_set()

    sess = tf.Session()

    if mode == 'gender':
        num_output = 2
        num_feat = 4096
    elif mode == 'age':
        num_output = 101
        num_feat = 101
    else:
        print("Unknown mode: " + mode)
        return False
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    """
    y = tf.placeholder(tf.float32, [None, 101])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    """

    vgg = vgg16(x, num_output, './VGGNet/vgg16_weights.npz', sess)


    # Evaluate model
    """
    correct_pred = tf.equal(tf.argmax(vgg.fc3l, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    """

    saver = tf.train.Saver()

    save_dir = './tf_model/'
    #load_path = './tf_model/tf_sex_1000.ckpt'
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if load_path is not None and ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, load_path)
        print("Loaded checkpoint")
    else:
        print("Checkpoint not found")

    features = np.empty((len(images), num_feat))
    for i in range(len(images)):
        print("Process %dth image: %s"%(i, images[i]))
        im = imread(images[i], mode='RGB')
        im = imresize(im, (224, 224))
        if mode == 'sex':
            features[i, :] = sess.run(vgg.fc2l, feed_dict={vgg.imgs: [im]})[0]
        else:
            features[i, :] = sess.run(vgg.fc3l, feed_dict={vgg.imgs: [im]})[0]
    pickle.dump(features, open(save_path, 'wb'))


if __name__ == '__main__':

    #  argv[1]: sex or training
    #  argv[2]: model path
    #  argv[3]: save features path
    main(sys.argv[1], sys.argv[2], sys.argv[3])

