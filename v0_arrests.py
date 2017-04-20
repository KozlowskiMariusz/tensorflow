
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

def data(name): return pandas2ri.ri2py(r[name])


df = data('USArrests')

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.1)

train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

train_x = train.ix[:,1:4] 
train_y = train['Murder']

test_x = test.ix[:,1:4]
test_y = test['Murder']

print(test_x.columns)

train_x = train_x.apply(lambda x: (x-np.mean(x)) / (np.max(x) - np.min(x)))
test_x = test_x.apply(lambda x: (x-np.mean(x)) / (np.max(x) - np.min(x)))

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 30

n_classes = 1
batch_size = 45 
hm_epochs = 5000


x = tf.placeholder('float')
y = tf.placeholder('float')

p = train_x.shape[1]
print(train_x.shape[0])
print(p)
print(train_x.shape)

hidden_1_layer = {'weight':tf.Variable(tf.random_normal([p, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])) }

def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.tanh(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.tanh(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.tanh(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']
    output = tf.transpose(output)
    return output

def train_neural_network(x):
    
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(prediction-y)))
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i+=batch_size
            if epoch % 1000 == 0: 
                print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        mae = tf.reduce_mean(tf.abs(prediction-y))

        pred_train = pd.DataFrame(prediction.eval({x:train_x, y:train_y}))
        pred_test = pd.DataFrame(prediction.eval({x:test_x, y:test_y}))
          
        pred_train = pred_train.transpose()
        pred_test = pred_test.transpose()       
 
        pred_train.columns = ['Murder_pred']
        train.reset_index(drop=True,inplace=True)
        pred_train['Murder'] = train['Murder']
        pred_train['error'] = pred_train['Murder_pred'] - pred_train['Murder']
        pred_train['abs_error'] = abs( pred_train['Murder_pred'] - pred_train['Murder'])
        mae_train=pred_train['abs_error'].mean()
        print('mae_train',mae_train)

      
        pred_test.columns = ['Murder_pred']
        test.reset_index(drop=True,inplace=True)
        pred_test['Murder'] = test['Murder']
        pred_test['error'] = pred_test.ix[:,0] - pred_test['Murder']
        pred_test['abs_error'] = abs( pred_test['Murder_pred'] - pred_test['Murder'])
        mae_test = pred_test['abs_error'].mean()
        print('mae_test', mae_test)
      
        pred_test.to_csv('arrestsTest.csv',sep=",",index=False)
        pred_train.to_csv('arrestsTrain.csv',sep=",",index=False)   
        print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        print('test_error:',mae.eval({x:test_x, y:test_y}))

	
train_neural_network(x)

