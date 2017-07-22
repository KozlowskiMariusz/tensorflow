# Tensorflow workshop with Michal Krason
# ---------------------------------------

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("data", one_hot=True)

hm_epochs = 3
n_classes = 10
batch_size = 100
chunk_size = 28
n_chunks = 28
rnn_size = 128

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder('float', [None, n_chunks, chunk_size])
    y = tf.placeholder('float')

    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    inp = tf.unstack(x, axis=1)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, inp, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for step in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
        print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))

    # Prepare RNN neural network for the Iris data.
    # Use:
    # - BasicLSTM cell
    # - Initialize data with random uniform distribution variables
    # - Put two rows of the picture as a chunk size
    # - RNN size should be 56
    # - Use SGD optimizer
