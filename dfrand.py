from rpy2.robjects import r
from rpy2.robjects import pandas2ri
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import random



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

train, test = train_test_split(df, test_size = 0.2)

train_x = train.ix[:,0:10] 
y_tr = np.array(train['y'])

test_x = test.ix[:,0:10]
y_te = np.array(test['y'])

train_y = []
test_y = []
for k in range(8000):
    train_y.append(y_tr[k])

for k in range(2000):
    test_y.append(y_te[k])




print(test_x.columns)


n_nodes_hl1 = 80
n_nodes_hl2 = 80
n_nodes_hl3 = 40
alpha = 0.01
n_classes = 1
batch_size = 5000
hm_epochs = 100000
hm_epochs_print = 1000
#print(len(train_x[0]))


x = tf.placeholder('float')
y = tf.placeholder('float')

p = train_x.shape[1]
print(train_x.shape[0])
print(p)
print(train_x.shape)


hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([p, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':1,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.sigmoid(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.losses.absolute_difference(labels=y, predictions=prediction ))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
           # train.sample(frac=1).reset_index(drop=True,inplace=True)
           # train_x = train.ix[:,0:10]
           # train_y = train['y']
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

        mae = tf.reduce_mean(tf.abs(prediction-y))

        pred_train = pd.DataFrame(prediction.eval({x:train_x, y:train_y}))
        pred_test = pd.DataFrame(prediction.eval({x:test_x, y:test_y}))
      
        pred_train.columns = ['y_pred']
       # print(pred_train.values) 
        train.reset_index(drop=True,inplace=True)
        pred_train['y'] = train['y']
        pred_train['error'] = pred_train['y_pred'] - pred_train['y']
        pred_train['abs_error'] = abs( pred_train['y_pred'] - pred_train['y'])
        mae_train=pred_train['abs_error'].mean()
        print('mae_train',mae_train)

      
        pred_test.columns = ['y_pred']
        test.reset_index(drop=True,inplace=True)
        pred_test['y'] = test['y']
        pred_test['error'] = pred_train.ix[:,0] - pred_train['y']
        pred_test['abs_error'] = abs( pred_train['y_pred'] - pred_train['y'])
        mae_test = pred_test['abs_error'].mean()
        print('mae_test', mae_test)
      
        pred_test.to_csv('dfrandTest.csv',sep=",",index=False)
        pred_train.to_csv('dfrandTrain.csv',sep=",",index=False)   
        print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        print('test_error:',mae.eval({x:test_x, y:test_y}))

	
train_neural_network(x)

