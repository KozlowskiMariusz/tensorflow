from rpy2.robjects import r
from rpy2.robjects import pandas2ri
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-layers',required=True, help = 'architecture of neural network', type=int, nargs='+')
parser.add_argument('-epochs', '--hm_epochs', help="number of epochs", type=int, default =3000)
parser.add_argument('-checkpoint', '--hm_epochs_print', action="store", help="number of epochs print - loss value checkpoint",type=int, default=1000)
parser.add_argument('-alpha', required=True, help="learning rate", type=float)
parser.add_argument('-batch', '--batch_size', action="store", help="batch size", type=int,  default=1000)

args = parser.parse_args()

# Creating random data aset
n = 10000
x1 = np.random.randn(n)
x2 = np.random.randn(n)
x3 = np.random.randn(n)
x4 = np.random.randn(n)
x5 = np.random.randn(n)
x6 = np.random.randn(n)
x7 = np.random.randn(n)
x8 = np.random.randn(n)
x9 = np.random.randn(n)
x10 = np.random.randn(n)
        

y0 = np.power(x1,4) + np.power(x2,2) + np.power(x3,2) + 11*x4 + 7*x5 + 5*x6 + x7 + 3*x8 + 2*x9 + x10 + 50

for k in range(n):
    if(y0[k] > 100):
        y0[k] = np.mean(y0) + np.random.randn(1)
    if(y0[k] <= 0):
        y0[k] = 5 + np.random.randn(1)

df = pd.DataFrame({'y':y0,'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5,'x6':x6,'x7':x7,'x8':x8,'x9':x9,'x10':x10})

# Dividing data frame into train and test set

train, test = train_test_split(df, test_size = 0.2)

train_x = train.ix[:,0:10] 
test_x = test.ix[:,0:10]

train_y = list(train['y'])
test_y = list(test['y'])

#defining arguments given by user
n_classes = 1 # continous output
alpha = args.alpha
batch_size = args.batch_size
hm_epochs = args.hm_epochs
hm_epochs_print = args.hm_epochs_print

p = train_x.shape[1]
print("train_x dimensions: ", train_x.shape)
layers = [p] + args.layers + [n_classes]
n_layers = len(layers)
print("layers: ", layers)


# TRAINING

x = tf.placeholder('float')
y = tf.placeholder('float')


def neural_network_model(data):
 
    last_hidden_layer = data
    for j in range(2,n_layers): 
        previous_layer = last_hidden_layer
        weight  = tf.Variable(tf.random_normal([layers[j-2], layers[j-1] ] )) 
        bias = tf.Variable(tf.random_normal([layers[j-1] ]))
        last_hidden_layer = tf.nn.sigmoid( tf.add(tf.matmul(previous_layer, weight), bias) )
        print("last_hidden_layer",last_hidden_layer)

    weight_output = tf.Variable(tf.random_normal([layers[-2], layers[-1]] ))
    bias_output =  tf.Variable(tf.random_normal([layers[-1]] ))
    output = tf.add(tf.matmul(last_hidden_layer, weight_output), bias_output)
    output = tf.transpose(output)

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(y-prediction))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            train.sample(frac=1).reset_index(drop=True,inplace=True)
            train_x = train.ix[:,0:10]
            train_y = train['y']
            _, epoch_loss = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i+=batch_size
            if epoch % hm_epochs_print == 0: 
                print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
#TESTING
        mae = tf.reduce_mean(tf.abs(prediction-y)) 
        print(mae.eval({x:train_x, y:train_y}))
        print(mae.eval({x:test_x, y:test_y}))

        pred_train = pd.DataFrame(prediction.eval({x:train_x, y:train_y}))
        pred_test = pd.DataFrame(prediction.eval({x:test_x, y:test_y}))
       
        pred_train = pred_train.transpose()
        pred_test = pred_test.transpose()
        
        pred_train.columns = ['y_pred']
        train.reset_index(drop=True,inplace=True)
        pred_train['y'] =  train['y']
        pred_train['error'] = pred_train['y_pred'] - pred_train['y']
        pred_train['abs_error'] = abs( pred_train['y_pred'] - pred_train['y'])
        mae_train=pred_train['abs_error'].mean()
        print('mae_train',mae_train)
      
        pred_test.columns = ['y_pred']
        test.reset_index(drop=True,inplace=True)
        pred_test['y'] = test['y']
        pred_test['error'] = pred_test.ix[:,0] - pred_test['y']
        pred_test['abs_error'] = abs( pred_test['y_pred'] - pred_test['y'])
        mae_test = pred_test['abs_error'].mean()
        print('mae_test', mae_test)
      
        pred_test.to_csv('dfrandTest.csv',sep=",",index=False)
        pred_train.to_csv('dfrandTrain.csv',sep=",",index=False)   

	
train_neural_network(x)


