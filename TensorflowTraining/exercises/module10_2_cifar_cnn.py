# Tensorflow workshop with Jan Idziak
# -------------------------------------
#
# script harvested from:
# https://pythonprogramming.net
#
# Implementing Convolutional Neural Network
# ---------------------------------------
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical
import numpy as np

#model_path = '/home/michal/DataScience/Data Scientist/Udacity/Deep Learning/Assignment1/'
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

n_classes = 10
batch_size = 100

graph = tf.Graph()

with graph.as_default():
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=2/np.sqrt(5*5*3))),
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=2/np.sqrt(3*3*32))),
               'W_fc': tf.Variable(tf.random_normal([8 * 8 * 64, 1024], stddev=2/np.sqrt(8*8*64*1024))),
               'out': tf.Variable(tf.random_normal([1024, n_classes], stddev=2/np.sqrt(1024*n_classes)))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.placeholder('float', [None, 32, 32, 3])
    y = tf.placeholder('float', [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv1'])
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv2'])
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc = tf.reshape(conv2, [-1, 8 * 8 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc, weights['out']) + biases['out']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

hm_epochs = 10
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for step in range(int(X.shape[0] / batch_size)):
            epoch_x = X[(step*batch_size):((step+1)*batch_size)]
            epoch_y = Y[(step * batch_size):((step + 1) * batch_size)]
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.8})
            epoch_loss += c
        print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
    acc = []
    for i in range(int(X_test.shape[0] / batch_size)):
        acc.append(accuracy.eval({x: X_test[(i*batch_size):((i+1)*batch_size)],
                                  y: Y_test[(i*batch_size):((i+1)*batch_size)], keep_prob: 1}))
    print('Accuracy:', sess.run(tf.reduce_mean(acc)))

# Prepare CNN neural network for the Iris data.
# Use:
# - Two conwolutional layers
# (filter size: first: 5, returns: 16, second: 3, returns:32)
# - Two pooling layers
# - reshape to appropriate size for the fully connected layer

# Train the network for 3 epochs
# Use bach size of 100
# Calculate accuracy for just 5000 first observations from train set
