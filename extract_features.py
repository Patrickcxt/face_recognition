import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle

model_dir = './inceptionv3'
#images_dir = '/home/amax/cxt/data/IMDB_WIKI/imdb_face_opencv/'
#images_dir = '/home/amax/cxt/data/IMDB_WIKI/imdb_data/'
images_dir = '/home/amax/cxt/data/IMDB_WIKI/wiki_face_opencv/'
list_images = [f for f in os.listdir(images_dir) if re.search('jpg', f)]
print(len(list_images))


def create_graph():
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    sex_labels = []
    age_labels = []
    create_graph()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #with tf.Session() as sess:
    #with tf.device("/gpu:1"):
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        k = 0
        for ind, image in enumerate(list_images):
            if (ind%10 == 0):
                print('{Processing %i th image: %s ..' % (ind, image))

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
            """
            if age_label < 0 :
                continue
            if age_label > 100:
                age_labels.append(10)
            else:
                age_labels.append(int(age_label/10))
            """


            image = images_dir + image
            if not gfile.Exists(image):
                tf.logging.fatal("File does not exist %s", image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
                    {'DecodeJpeg/contents:0': image_data})
            features[k,:] = np.squeeze(predictions)
            k = k + 1
        features = features[0:k]
    return features, sex_labels, age_labels


features, sex_labels, age_labels = extract_features(list_images)

pickle.dump(features, open('./wiki_opencv_features/features_precise', 'wb'))
pickle.dump(sex_labels, open('./wiki_opencv_features/sex_labels_precise', 'wb'))
pickle.dump(age_labels, open('./wiki_opencv_features/age_labels_precise', 'wb'))

