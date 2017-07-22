import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

n_nodes_hl1 = 50
n_nodes_hl2 = 20
n_nodes_hl3 = 10

n_classes = 3

""" Read the iris data set and split them into training and test sets """
iris = datasets.load_iris()
data = iris["data"]
target = iris["target"]

# Convert into one-hot vectors
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!
train_X, test_X, train_y, test_y = train_test_split(data, all_Y, test_size=0.33, random_state=RANDOM_SEED)

x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
y_size = train_y.shape[1]

graph = tf.Graph()

with graph.as_default():
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([x_size, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(X, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    acc = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    # Run SGD
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(15):
            # Train with each example
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * acc.eval({X: train_X, y: train_y}),
                     100. * acc.eval({X: test_X, y: test_y})))

            # predict = tf.argmax(output, axis=1)
            # train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
            #                          sess.run(predict, feed_dict={X: train_X, y: train_y}))
            # test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
            #                         sess.run(predict, feed_dict={X: test_X, y: test_y}))
