# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import math as math
# import data using pandas
pdtrain = pd.read_csv('train.csv')
pdtest = pd.read_csv('test.csv')

# na
pdtrain.Age = pdtrain.Age.fillna(pdtrain.Age.mean())
pdtrain.Cabin = pdtrain.Cabin.fillna('')
pdtrain.Embarked = pdtrain.Embarked.fillna('')

train = tf.contrib.learn.extract_pandas_matrix(pdtrain)
test = tf.contrib.learn.extract_pandas_matrix(pdtest)

# split training set into training/cross validation set
# use 70/30
trainLength = len(train)
testLength = len(test)
training_set_length = int(math.floor(trainLength * 0.7))
training_set = train[:training_set_length]
crossval_set = train[training_set_length:,:]

# spare data
Sex = tf.contrib.layers.sparse_column_with_keys(column_name='Sex', keys=['female','male'])
Embarked = tf.contrib.layers.sparse_column_with_keys(column_name='Embarked', keys=['S','Q','C'])
Cabin = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='Cabin', hash_bucket_size=1000)
Name = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='Name', hash_bucket_size=1000)
Ticket = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='Ticket', hash_bucket_size=1000)
SibSp = tf.contrib.layers.sparse_column_with_integerized_feature(column_name='SibSp', bucket_size=100, dtype=tf.int8)
Pclass = tf.contrib.layers.sparse_column_with_integerized_feature(column_name='Pclass', bucket_size=5, dtype=tf.int8)
Parch = tf.contrib.layers.sparse_column_with_integerized_feature(column_name='Fare', bucket_size=10, dtype=tf.int8)

# continuous data
PassengerId = tf.contrib.layers.real_valued_column('PassengerId')
Age = tf.contrib.layers.real_valued_column('Age')
Fare = tf.contrib.layers.real_valued_column('Fare')

# wide columns
wide_columns = [Sex, Embarked, Cabin, Name, Ticket,SibSp, Pclass, Parch]

# deep columns
deep_columns = [PassengerId, Age, Fare,
        tf.contrib.layers.embedding_column(Pclass,dimension=8),
        tf.contrib.layers.embedding_column(Parch,dimension=8),
        tf.contrib.layers.embedding_column(SibSp,dimension=8),
        tf.contrib.layers.embedding_column(Sex,dimension=8)]

import tempfile
model_dir = tempfile.mkdtemp()
model = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = [100, 50])

# define columns
COLUMNS = ['PassengerId','Pclass','Name','Sex','Age','SibSp', 'Parch', 'Ticket', 'Fare','Cabin','Embarked']
LABEL_COLUMN = 'Survived'

CATEGORICAL_COLUMNS = ['Sex', 'Embarked','Cabin','Name','Ticket','SibSp','Pclass','Parch']
CONTINUOUS_COLUMNS = ['PassengerId','Age','Fare']

df_train = pdtrain[:training_set_length]
df_test = pdtrain[training_set_length:]

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
    indices=[[i, 0] for i in range(df[k].size)],
    values=df[k].values,
    shape=[df[k].size, 1]
    )
    for k in CATEGORICAL_COLUMNS}                  # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

model.fit(input_fn=train_input_fn, steps=200)
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])
