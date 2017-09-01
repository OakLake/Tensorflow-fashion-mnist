# fashion_MNIST using TensorFlow
from __future__ import print_function

import numpy as np
import tensorflow as tf
from mnist_reader import load_mnist
import os
import matplotlib.pyplot as plt
import matplotlib.image  as img

os.system('clear')
print('Library import complete.')

# loading the data
path = './data/fashion'
train_images,train_labels = load_mnist(path, kind='train')
test_images,test_labels = load_mnist(path, kind='t10k')

print('Data loading complete.')

# reshaping data
image_size = 28
num_channels = 1
num_labels = 10


train_images = train_images.reshape(-1,image_size,image_size,num_channels).astype(np.float32)
test_images = test_images.reshape(-1,image_size,image_size,num_channels).astype(np.float32)

train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
#train_labels.reshape(-1,1)
test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)
#test_labels.reshape(-1,1)

# train_labels = tf.one_hot(train_labels,10)
# test_labels = tf.one_hot(test_labels,10)

print('Train dataset',train_images.shape)
print('Train labels',train_labels.shape)

print('Test dataset',test_images.shape)
print('Test labels',test_labels.shape)


# plt.imshow(train_images[0].reshape(28,28) ,cmap='gray')
# plt.show()


# TensorFlow - CNN
graph = tf.Graph()

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
initial_alpha = 1e-3
alpha_decay = 0.97

with graph.as_default():

    # Input data .
    tf_train_data = tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size,num_channels])
    #tf.constant(train_images)
    tf_train_labels = tf.placeholder(tf.float32,shape=[batch_size,num_labels])
    #tf.constant(train_labels)
    tf_test_data = tf.constant(test_images)
    tf_test_labels = tf.constant(test_labels)

    # Variables .
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(initial_alpha, global_step, 2001, alpha_decay, True)

    layer1_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev = 0.1))
    layer2_biases = tf.Variable(tf.constant(1.0,shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size//4 * image_size//4 * depth,num_hidden],stddev = 0.1))
    layer3_biases = tf.Variable(tf.constant(1.0,shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0,shape=[num_labels]))


    # Model .
    def model(dataset):
        conv = tf.nn.conv2d(dataset,layer1_weights,[1,2,2,1],padding = 'SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden,layer2_weights,[1,2,2,1],padding = 'SAME')
        hidden = tf.nn.relu(conv + layer2_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape,layer3_weights) + layer3_biases)
        return tf.matmul(hidden,layer4_weights) + layer4_biases

    logits = model(tf_train_data)

    # loss .
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,logits = logits))

    # optimizer

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # predictions .
    train_predictions = tf.nn.softmax(model(tf_train_data))
    test_predictions = tf.nn.softmax(model(tf_test_data))


def accuracy(predictions, labels):
    # return tf.metrics.accuracy(labels,predictions)
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

num_steps = 2001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('\n\nInitialized\n\n')

    for step in range(num_steps):
        # print('step: ',step)

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}

        _,l,predictions = session.run([optimizer,loss,train_predictions],feed_dict=feed_dict)
        if step % 100 == 0:
            print('...............................................')
            print('Minibatch Train loss at step %d : %f' % (step,l))
            print('Minibatch Train accuracy: %.1f%%' % accuracy(predictions,batch_labels))

    print('***************')
    print('Test accuracy: %.1f%%' % accuracy(test_predictions.eval(), test_labels))







# eop
