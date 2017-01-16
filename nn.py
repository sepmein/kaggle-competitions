from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

TRAINING = 'train_new.csv'
TEST = 'crossval.csv'

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename = TRAINING,
        target_dtype = np.int,
        features_dtype = np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename = TEST,
        target_dtype = np.int,
        features_dtype = np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("",dimension=8)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units = [10,20,10], n_classes = 2, model_dir ="./model")
classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

import csv as csv
test_file = csv.reader(open('./test_applied.csv','rb'))
testdata = []
for row in test_file:
    testdata.append(row)
testdatanp = np.array(testdata)
y = testdatanp.astype(np.float)
print(y)
