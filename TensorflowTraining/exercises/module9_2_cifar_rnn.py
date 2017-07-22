# Tensorflow workshop with Michal Krason
# ---------------------------------------

import tensorflow as tf
from tensorflow.contrib import rnn
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical
import numpy as np

(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

hm_epochs = 8
n_classes = 10
batch_size = 100
chunk_size = 32 * 3
n_chunks = 32
rnn_size = 128

graph = tf.Graph()

with graph.as_default():
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.placeholder('float', [None, 32, 32, 3])
    y = tf.placeholder('float', [None, 10])
    inp = tf.reshape(x, [-1, n_chunks, chunk_size])
    inp = tf.unstack(inp, axis=1)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, inp, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        for step in range(int(X.shape[0] / batch_size)):
            epoch_x = X[(step * batch_size):((step + 1) * batch_size)]
            epoch_y = Y[(step * batch_size):((step + 1) * batch_size)]
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
        print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
    acc = []
    for i in range(int(X_test.shape[0] / batch_size)):
        acc.append(accuracy.eval({x: X_test[(i * batch_size):((i + 1) * batch_size)],
                                  y: Y_test[(i * batch_size):((i + 1) * batch_size)]}))
    print('Accuracy:', sess.run(tf.reduce_mean(acc)))

    # Prepare RNN neural network for the Iris data.
    # Use:
    # - BasicLSTM cell
    # - Initialize data with random uniform distribution variables
    # - Put two rows of the picture as a chunk size
    # - RNN size should be 56
    # - Use SGD optimizer
