# Module 4: Simple TF Model
# Use softmax cross-entropy function with MINST Dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
yhat = tf.matmul(X, W) + b
y = tf.placeholder(tf.float32, [None, 10]) # placeholder for correct answers

# Step 3: Loss Functions
loss = tf.reduce_mean(
   tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# % of correct answer found in batches
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for i in range(1000):
    batch_X, batch_y = mnist.train.next_batch(100)
    train_data = {X: batch_X, y: batch_y}
    sess.run(train, feed_dict=train_data)
    print("Training Accuracy = ",sess.run(accuracy,feed_dict = train_data))

# Step 5: Prediction
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))