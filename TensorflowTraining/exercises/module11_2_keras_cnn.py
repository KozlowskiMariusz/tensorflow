# Tensorflow workshop with Jan Idziak
# -------------------------------------
#
# keras CNN model for the mnist data set based on the kaggle script:
# https://www.kaggle.com/aman11dhanpat/digit-recognizer
# ------------------------------------------------------
#
from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_yaml

batch_size = 56
nb_classes = 10
nb_epoch = 1
mnist = input_data.read_data_sets("data", one_hot=True)
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size (filter size for CNN)
kernel_size = (3, 3)

X_train = mnist.train.images
X_test = mnist.test.images

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
Y_train = mnist.train.labels
Y_test = mnist.test.labels
input_shape = (img_rows, img_cols, 1)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size=(5,5),
                        padding='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Starting training")
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, Y_test),
          verbose=1)
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
model.predict(X_test, batch_size=batch_size, verbose=1)

# model - save weights only per name and load to named layers


# Prepare CNN using KERAS
# - copy the architecture from the Module 7_1
# - use siftsign and relu6 activation functions where required
# - make sure filter sizes are appropriate
# - do not use dropout for this case
