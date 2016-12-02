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
#from imagenet_classes import class_names
from vgg16 import vgg16

def get_dataset(images_dir):
    print("Get training set ...")
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

    # convert age_label to one-hot target
    targets = np.array(age_labels)
    ohm = np.zeros((targets.shape[0],101))
    ohm[np.arange(targets.shape[0]), targets] = 1
    age_labels = ohm

    # convert sex_label to one-hot target
    targets = np.array(sex_labels)
    ohm = np.zeros((targets.shape[0], 2))
    ohm[np.arange(targets.shape[0]), targets] = 1
    sex_labels = ohm

    return images, sex_labels, age_labels


#parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 5
train = True

def main(mode):
    # Get dataset
    """
    # Fine tune using imdb
    images_dir = '/home/amax/cxt/data/IMDB_WIKI/imdb_face_opencv/'
    train_images, train_sex_labels, train_age_labels = get_dataset(images_dir)
    images_dir = '/home/amax/cxt/data/IMDB_WIKI/wiki_face_opencv/'
    test_images, test_sex_labels, test_age_labels = get_dataset(images_dir)
    """

    # Fine-tune using wiki-opencv training set
    images_dir = '/home/amax/cxt/data/IMDB_WIKI/wiki_face_opencv/'
    images, sex_labels, age_labels = get_dataset(images_dir)
    train_images = images[:22000]
    test_images = images[22000:]
    train_sex_labels = sex_labels[:22000]
    test_sex_labels = sex_labels[22000:]

    if mode == 'gender':
        num_output = 2
        train_labels = sex_labels[:22000]
        test_labels = sex_labels[22000:]
    elif mode == 'age':
        num_output = 101
        train_labels = age_labels[:22000]
        test_labels = age_labels[22000:]
    else:
        print("Unkonw mode: " + mode)
        return False

    sess = tf.Session()
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, num_output])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    vgg = vgg16(x, num_output, './VGGNet/vgg16_weights.npz', sess)

    # Define loss and optimizer
    cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.fc3l, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(vgg.fc3l, 1), tf.argmax(y, 1))

    if sys.argv[1] == 'sex':
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    else:
        accuracy = tf.reduce_mean(tf.square(tf.argmax(vgg.fc3l, 1) - tf.argmax(y, 1)))

    saver = tf.train.Saver()

    if train:
        # load saved model
        """
        save_dir = './tf_model/'
        load_path = './tf_model/tf_age_3000.ckpt'
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if load_path is not None and ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, load_path)
            print("Loaded checkpoint")
        else:
            print("Checkpoint not found")
        """

        perm = np.random.permutation(len(train_images))  # shuffle
        step, it = 1, 0

        while it + batch_size -1 < len(train_images):

            # Get Minibatch
            batch_index = perm[it:it+batch_size]
            batch_im = []
            batch_labels = []
            for ind in batch_index:
                im = imread(train_images[ind], mode='RGB')
                im = imresize(im, (224, 224))
                batch_im.append(im)
                batch_labels.append(train_labels[ind])


            sess.run(optimizer, feed_dict={x: batch_im, y: batch_labels, keep_prob: 0.5})

            if step % display_step  == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_im,
                                                                  y: batch_labels,
                                                                  keep_prob: 1.0})
                print("\nEpoch " + str(step) + " Iter " + str(it) + ", Minibatch Loss= " +  \
                        "{:.6f}".format(loss) + ", Training Accuracy = " + \
                        "{:.5f}".format(acc))

            if step % 100 == 0:
                Acc = 0.0
                for ind in range(100):
                    im = imread(test_images[ind], mode='RGB')
                    im = imresize(im, (224, 224))
                    acc = sess.run(accuracy, feed_dict={x: [im],
                                                          y: [test_labels[ind]],
                                                          keep_prob: 1.0})
                    Acc += acc
                Acc = Acc / 100.0
                print("\n==================================")
                print("Epoch " + str(step) + ", Test Accuracy = " + \
                        "{:.5f}".format(Acc))
                print("==================================\n")

            if step % 1000 == 0:
                checkpoint_dir = './tf_model/tf_' + sys.argv[1] + '_' + str(step) + '.ckpt'
                saver.save(sess, checkpoint_dir)
                print("\n==================================")
                print("Model " + checkpoint_dir + " saved")
                print("==================================\n")

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
        load_path = './tf_model/tf_age_5000.ckpt'
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if load_path is not None and ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, load_path)
            print("Loaded checkpoint")

            Acc = 0.0
            for ind in range(len(test_images[:100])):
                im = imread(test_images[ind], mode='RGB')
                im = imresize(im, (224, 224))
                acc, pred = sess.run([accuracy, vgg.probs] , feed_dict={x: [im],
                                                    y: [test_labels[ind]],
                                                    keep_prob: 1.0})
                """
                print(acc)
                print(np.argmax(pred, axis=1)[0], np.argmax(test_age_labels[ind]))
                """
                Acc += acc
            Acc = Acc / len(test_images[:100])
            print("Test Accuracy = {:.5f}".format(Acc))
        else:
            print("No checkpoint found...")
            return False

        print("Evaluation Finished!")

    return True



if __name__ == '__main__':

    main(sys.argv[1])




