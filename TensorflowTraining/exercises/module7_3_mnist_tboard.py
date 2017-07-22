import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot = True)

n_nodes_hl1 = 500

n_classes = 10
batch_size = 100

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')
    logs_path = '/tmp/tensorflow_logs_challenge/example'
    with tf.name_scope('Variables'):
        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                              'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

        output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                            'biases':tf.Variable(tf.random_normal([n_classes])),}

    with tf.name_scope('Model'):
        l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']
    with tf.name_scope('Loss'):
    # Minimize error using cross entropy
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y) )
    with tf.name_scope('SGD'):
    # Gradient Descent    
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.name_scope('Accuracy'):
    # Accuracy
        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    
hm_epochs = 10
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, graph=graph)
    for epoch in range(hm_epochs):
        epoch_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: epoch_x, y: epoch_y})
            summary_writer.add_summary(summary, epoch * total_batch + i)
            epoch_loss += c

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

    print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

print("Run the command line:\n" \
      "--> tensorboard --logdir=/tmp/tensorflow_logs_chalenge " \
      "\nThen open http://127.0.1.1:6006/ into your web browser")

