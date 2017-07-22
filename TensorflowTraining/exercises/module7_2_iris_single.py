import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

n_nodes_hl1 = 100
n_classes = 3
batch_size = 100
""" Read the iris data set and split them into training and test sets """
iris   = datasets.load_iris()
data   = iris["data"]
target = iris["target"]

# Convert into one-hot vectors
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!
train_X, test_X, train_y, test_y = train_test_split(data, all_Y, test_size=0.33, random_state=RANDOM_SEED)

x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
y_size = train_y.shape[1]

X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None, y_size])


hidden_1_layer = {'weights': tf.Variable(tf.random_normal([4, n_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}

l1 = tf.add(tf.matmul(X, hidden_1_layer['weights']), hidden_1_layer['biases'])
l1 = tf.nn.relu(l1)

output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

#predict = tf.argmax(output, axis=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(70):
        # Train with each example
        epoch_loss = 0
        for i in range(int(len(train_X)/batch_size)):
            _, c = sess.run([updates, cost], feed_dict={X: train_X[i*batch_size: (i + 1)*batch_size], y: train_y[i*batch_size: (i + 1)*batch_size]})
            epoch_loss += c
        # train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
        #                          sess.run(predict, feed_dict={X: train_X, y: train_y}))
        # test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
        #                          sess.run(predict, feed_dict={X: test_X, y: test_y}))
        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * accuracy.eval(feed_dict={X: train_X, y: train_y}), 100. * accuracy.eval(feed_dict={X: test_X, y: test_y})))
