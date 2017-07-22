# Module 4: Simple TF Model
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Step 1: Initial Setup
X = tf.placeholder(tf.float32)
W = tf.Variable([0.1],tf.float32)
b = tf.Variable([0.1],tf.float32)

# Step 2: Model
yhat = tf.multiply(W,X) + b
y = tf.placeholder(tf.float32) # Placeholder for correct answer

# # Step 3: Loss Function
loss = tf.reduce_sum(tf.square(yhat - y)) # sum of the squares error

# # Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# # training data
train_X = [1,2,3,4]
train_y = [0,-1,-2,-3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# # Step 5: Training Loop
for i in range(1000):
  sess.run(train, {X:train_X, y:train_y})

# Step 6: Evaluation
import matplotlib.pyplot as plt
plt.plot(train_X,train_y,'o')
plt.plot(train_X,sess.run(tf.multiply(W,train_X)+b),'r')
plt.show()

