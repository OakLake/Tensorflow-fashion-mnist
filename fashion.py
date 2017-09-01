# fashion_MNIST using TensorFlow
from __future__ import print_function

import numpy as np
import tensorflow as tf
from mnist_reader import load_mnist
import os
import matplotlib.pyplot as plt
os.system('clear')
print('Library import complete.')

# loading the data
path = './data/fashion'
train_images,train_labels = load_mnist(path, kind='train')
test_images,test_labels = load_mnist(path, kind='t10k')

print('Data loading complete.')

print('Train dataset',train_images.shape)
print('Train labels',train_labels.shape)

print('Test dataset',test_images.shape)
print('Test labels',test_labels.shape)
