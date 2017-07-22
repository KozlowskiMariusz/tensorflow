import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
n_nodes_hl1 = 100

n_classes = 3
logs_path = '/tmp/tensorflow_logs_challenge/example'

""" Read the iris data set and split them into training and test sets """
iris = datasets.load_iris()
data = iris["data"]
target = iris["target"]

# Convert into one-hot vectors
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!
train_X, test_X, train_y, test_y = train_test_split(data, all_Y, test_size=0.33, random_state=RANDOM_SEED)

x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
y_size = train_y.shape[1]

X = tf.placeholder(tf.float32, shape=[None, x_size], name='Input_data')
y = tf.placeholder(tf.float32, shape=[None, y_size], name='Labels')

with tf.name_scope('parameters'):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([x_size, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}

with tf.name_scope('model'):
    l1 = tf.add(tf.matmul(X,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']

yhat = output
predict = tf.argmax(yhat, axis=1)

# Backward propagation
with tf.name_scope('loss'):
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
with tf.name_scope('SGD'):
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(predict, tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

tf.summary.scalar("Loss",cost)
tf.summary.scalar("Acc",acc)
merged_summary_op = tf.summary.merge_all()

# Run SGD
sess = tf.Session()
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(70):
    # Train with each example
    epoch_loss = 0
    for i in range(len(train_X)):
        _, c, summary = sess.run([updates, cost, merged_summary_op], feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
        summary_writer.add_summary(summary, global_step=epoch*len(train_X)+i)
        epoch_loss += c
    train_accuracy = acc.eval(session=sess, feed_dict={X: train_X, y: train_y})
    test_accuracy  = acc.eval(session=sess, feed_dict={X: test_X, y: test_y})
    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%, loss = %.1f"
          % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy, epoch_loss))

sess.close()

print("Run the command line:\n" \
      "--> tensorboard --logdir=/tmp/tensorflow_logs_challenge " \
      "\nThen open http://127.0.1.1:6006/ into your web browser")