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

train_images = train_images.reshape(-1,image_size,image_size,num_channels)
test_images = test_images.reshape(-1,image_size,image_size,num_channels)

train_labels = train_labels.reshape(-1,1)
test_labels = test_labels.reshape(-1,1)

print('Train dataset',train_images.shape)
print('Train labels',train_labels.shape)

print('Test dataset',test_images.shape)
print('Test labels',test_labels.shape)

plt.imshow(train_images[0].reshape(28,28) ,cmap='gray')
plt.show()


# TensorFlow - CNN
graph = tf.Graph()

patch_size = 5
depth = 16
num_hidden = 64
num_labels = 10

with graph.as_default():

    # Input data .
    tf_train_data = tf.constant(train_images)
    tf_train_labels = tf.constant(train_labels)
    tf_test_data = tf.constant(test_images)
    tf_test_labels = tf.constant(test_labels)

    # Variables .
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev = 0.1))
    layer2_biases = tf.Variable(tf.constant(1.,[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size//4 * image_size//4 * depth,num_hidden],stddev = 0.1))
    layer3_biases = tf.Variable(tf.constant(1.,[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.,[num_labels]))


    # Model .
    def model(dataset):
        conv = tf.nn.conv2d(dataset,layer1_weights,[1,2,2,1],padding = 'SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden,layer2_weights,[1,2,2,1],padding = 'SAME')
        hidden = tf.nn.relu(conv + layer2_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden,shape[0],shape[1]*shape[2]*shape[3])

        hidden = tf.nn.relu(tf.matmul(reshape,layer3_weights) + layer3_biases)
        return tf.matmul(hidden,layer4_weights) + layer4_biases

    logits = model(tf_train_data)

    # loss .
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,logits = logits))

    # optimizer

    optimizer = tf.GradientDescentOptimizer(0.5).minimize(loss)

    # predictions .
