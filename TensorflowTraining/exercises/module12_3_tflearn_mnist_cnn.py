# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG model to retrain
network for a new task (your own dataset).All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).

Using pretrained model for further training with other inputs.
There are several approaches to fine tuning - this is one of them.
'''
#from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

mnist = read_data_sets("data", one_hot = True)

X = mnist.train.images
Y = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels
X = np.reshape(X, newshape=(-1, 28, 28, 1))
X_test = np.reshape(X_test, newshape=(-1, 28, 28, 1))

num_classes = Y.shape[1]

network = input_data(shape=[None, 28, 28, 1])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.75) 
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
# Finetuning Softmax layer (Setting restore=False to not restore its weights)
softmax = fully_connected(network, num_classes, activation='softmax', restore=False)
regression = regression(softmax, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)  

model = tflearn.DNN(regression, checkpoint_path='mnist_apply',
                    max_checkpoints=3, tensorboard_verbose=0)
# Load pre-existing model, restoring all weights, except softmax layer ones
model.load('./models/mnist_apply_1')

# Start finetuning
model.fit(X, Y, n_epoch=1, validation_set=(X_test, Y_test), shuffle=True, 
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='mnist_apply')

model.save('./models/mnist_apply_1')