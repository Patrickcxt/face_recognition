import os
import re

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
#from imagenet_classes import class_names


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        print("Length: ", len(self.parameters))
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i == 30:
                sess.run(tf.initialize_variables(self.parameters[i:i+2]))
                break
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

    """
    def save_weights(self, weight_file):
        module_names = ["conv1_1_W", "conv1_1_b", "conv1_2_W", "conv1_2_b",
                        "conv2_1_W", "conv2_1_b", "conv2_2_W", "conv2_2_b",
                        "conv3_1_W", "conv3_1_b", "conv3_2_W", "conv3_2_b", "conv3_3_W", "conv3_3_b",
                        "conv4_1_W", "conv4_1_b", "conv4_2_W", "conv4_2_b", "conv4_3_W", "conv4_3_b",
                        "conv5_1_W", "conv5_1_b", "conv5_2_W", "conv5_2_b", "conv5_3_W", "conv5_3_b",
                        "fc6_W", "fc6_b",
                        "fc7_W", "fc7_b",
                        "fc8_W", "fc8_b"]
        weights = {}
        for i in range(len(module_names)):
            weights[module_names[i]] = self.parameters[i]
        print(weights.keys())
        raw_input()
        print(weights["fc8_b"])
        np.save(weight_file, weights)
    """


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


#parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 5
train = True

if __name__ == '__main__':

    # Get dataset
    images, sex_labels, age_labels = get_train_set()
    train_images = images[:22000]
    test_images = images[22000:]
    train_age_labels = age_labels[:22000]
    test_age_labels = age_labels[22000:]

    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    vgg = vgg16(x, './VGGNet/vgg16_weights.npz', sess)


    # Define loss and optimizer
    cost  = tf.reduce_mean(tr.square(vgg.fc3l - y))
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Evaluate model
    """
    correct_pred = tf.equal(tf.argmax(vgg.fc3l, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    """

    saver = tf.train.Saver()

    if train:
        perm = np.random.permutation(len(train_images))
        step, it = 1, 0

        while it + batch_size -1 < len(train_images):

            # Get Minibatch
            batch_index = perm[it:it+batch_size]
            batch_im = []
            batch_age_labels = []
            for ind in batch_index:
                im = imread(train_images[ind], mode='RGB')
                im = imresize(im, (224, 224))
                batch_im.append(im)
                batch_age_labels.append(train_age_labels[ind])


            _, c = sess.run([optimizer, cost], feed_dict={x: batch_im, y: batch_age_labels, keep_prob: 0.5})

            print("\nEpoch " + str(step) + " Iter " + str(it) + ", Minibatch Loss= " +  \
                        "{:.6f}".format(c))

            """
            if step % 100 == 0:
                Acc = 0.0
                for ind in range(50):
                    im = imread(test_images[ind], mode='RGB')
                    im = imresize(im, (224, 224))
                    acc = sess.run(accuracy, feed_dict={x: [im],
                                                          y: [test_age_labels[ind]],
                                                          keep_prob: 1.0})
                    Acc += acc
                Acc = Acc / 50.0
                print("\n==================================")
                print("Epoch " + str(step) + ", Test Accuracy = " + \
                        "{:.5f}".format(Acc))
                print("==================================\n")

            if step % 2000 == 0:
                checkpoint_dir = './tf_model/tf_age_' + str(step) + '.ckpt'
                saver.save(sess, checkpoint_dir)
                print("\n==================================")
                print("Model " + checkpoint_dir + " saved")
                print("==================================\n")
            """

            step += 1
            it += batch_size
            if it + batch_size - 1 >= len(train_images):
                perm = np.random.permutation(len(train_images))
                it = 0
        print("Optimization Finished!")

    else:
        # Evaluate
        # Load model
        save_dir = './tf_model/'
        load_path = './tf_model/tf_age_5.ckpt'
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if load_path is not None and ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, load_path)
            print("Loaded checkpoint")

            Acc = 0.0
            for ind in range(len(test_images[:20])):
                im = imread(test_images[ind], mode='RGB')
                im = imresize(im, (224, 224))
                acc = sess.run(accuracy, feed_dict={x: [im],
                                                    y: [test_age_labels[ind]],
                                                    keep_prob: 1.0})
                print(acc)
                Acc += acc
            Acc = Acc / len(test_images[:20])
            print("Test Accuracy = {:.5f}".format(Acc))
        else:
            print("No checkpoint found...")

        print("Evaluation Finished!")









