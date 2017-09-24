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
# fashion-MNIST : './data/fashion'
# MNIST : './data/mnist'
path = './data/mnist'
train_images,train_labels = load_mnist(path, kind='train')
test_images_orig,test_labels_orig = load_mnist(path, kind='t10k')

# defining category dictionary
cat_dict = {0:'T-shirt/Top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle Boot'}

# splitting t10k into validation and test sets 5k each
test_images = test_images_orig[:5000]
test_labels = test_labels_orig[:5000]
valid_images = test_images_orig[5000:]
valid_labels = test_labels_orig[5000:]

print('Data loading complete.')

# values
image_size = 28
num_channels = 1
num_labels = 10


# reshaping for tf, and normalisation
train_images = train_images.reshape(-1,image_size,image_size,num_channels).astype(np.float32)
valid_images = valid_images.reshape(-1,image_size,image_size,num_channels).astype(np.float32)
test_images = test_images.reshape(-1,image_size,image_size,num_channels).astype(np.float32)

# one hot encoding
train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)

# printing sizes for check
print('Train dataset',train_images.shape)
print('Train labels',train_labels.shape)
print('Valid dataset',valid_images.shape)
print('Valid labels',valid_labels.shape)
print('Test dataset',test_images.shape)
print('Test labels',test_labels.shape)

# show random image from training set
r_ix = np.random.randint(6e3)
label = np.argwhere(train_labels[r_ix] == 1.)[0][0]
print('item is: ',cat_dict[label])
plt.imshow(train_images[r_ix].reshape(28,28) ,cmap='gray')
plt.show()

# TensorFlow - CNN
graph = tf.Graph()

batch_size = 16
patch_size = 5
depth = 32
depth2 = 64
depth3 = 128
num_hidden = 128
initial_alpha = 1e-3
alpha_decay = 0.97
dropout = 0.8

with graph.as_default():

    # Input data .
    tf_train_data = tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size,num_channels])
    tf_train_labels = tf.placeholder(tf.float32,shape=[batch_size,num_labels])

    tf_valid_data = tf.constant(valid_images)
    # tf_valid_labels = tf.constant(valid_labels)

    tf_test_data = tf.constant(test_images)
    # tf_test_labels = tf.constant(test_labels)


    # Variables .
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(initial_alpha, global_step, 15001, alpha_decay, True)

    layer1_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth2],stddev = 0.1))
    layer2_biases = tf.Variable(tf.constant(1.0,shape=[depth2]))

    # layer2b_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,depth2,depth3],stddev = 0.1))
    # layer2b_biases = tf.Variable(tf.constant(1.0,shape=[depth3]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size//4 * image_size//4 * depth2,num_hidden],stddev = 0.1))
    layer3_biases = tf.Variable(tf.constant(1.0,shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0,shape=[num_labels]))


    # Model .
    def model(dataset):
        conv = tf.nn.conv2d(dataset,layer1_weights,[1,1,1,1],padding = 'SAME')
        pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding='SAME')
        hidden = tf.nn.relu(pool + layer1_biases)
        conv = tf.nn.conv2d(hidden,layer2_weights,[1,1,1,1],padding = 'SAME')
        pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding='SAME')
        hidden = tf.nn.relu(pool + layer2_biases)

        # conv = tf.nn.conv2d(hidden,layer2b_weights,[1,1,1,1],padding = 'SAME')
        # pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding='SAME')
        # hidden = tf.nn.relu(pool + layer2b_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])
        # print(reshape.shape)
        # print(layer3_weights.shape)
        hidden = tf.nn.relu(tf.matmul(reshape,layer3_weights) + layer3_biases)
        return tf.matmul(hidden,layer4_weights) + layer4_biases

    def model_DO(dataset):
        conv = tf.nn.conv2d(dataset,layer1_weights,[1,1,1,1],padding = 'SAME')
        pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding='SAME')
        hidden = tf.nn.relu(pool + layer1_biases)
        conv = tf.nn.conv2d(hidden,layer2_weights,[1,1,1,1],padding = 'SAME')
        pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding='SAME')
        hidden = tf.nn.relu(pool + layer2_biases)

        # conv = tf.nn.conv2d(hidden,layer2b_weights,[1,1,1,1],padding = 'SAME')
        # pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding='SAME')
        # hidden = tf.nn.relu(pool + layer2b_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])
        # print(reshape.shape)
        # print(layer3_weights.shape)
        hidden = tf.nn.relu(tf.matmul(reshape,tf.nn.dropout(layer3_weights,dropout)) + layer3_biases)
        return tf.matmul(hidden,layer4_weights) + layer4_biases

    logits = model_DO(tf_train_data)

    # loss .
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,logits = logits)) + tf.nn.l2_loss(layer3_weights)

    # optimizermax

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # predictions .
    train_predictions = tf.nn.softmax(model(tf_train_data))
    valid_predictions = tf.nn.softmax(model(tf_valid_data))
    test_predictions = tf.nn.softmax(model(tf_test_data))


def accuracy(predictions, labels):
    # return tf.metrics.accuracy(labels,predictions)
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

num_steps = 20001

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
            print('_________ Valid accuracy: %.1f%% v' % accuracy(valid_predictions.eval(),valid_labels))
    print('***************')
    #print('Train accuracy: %.1f%%' % accuracy(train_predictions.eval(), train_labels)) # invalid batch(data) shape without the code inside the for loop
    print('Valid accuracy: %.1f%%' % accuracy(valid_predictions.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_predictions.eval(), test_labels))







# eop
