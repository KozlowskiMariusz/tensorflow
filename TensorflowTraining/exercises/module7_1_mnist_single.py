# Tensorflow workshop with Jan Idziak
# -------------------------------------
#
# script  based on :
# https://pythonprogramming.net
#
# Implementing single-layer Neural Network
# ---------------------------------------
#
# We will illustrate how to create a one hidden layer NN


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data", one_hot=True)

n_nodes_hl1 = 500

n_classes = 10
batch_size = 100

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

hm_epochs = 10
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        for i in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


# Prepare one layer perceptron neural network for the Iris data.