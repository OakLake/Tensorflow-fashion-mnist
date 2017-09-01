# fashion_MNIST using TensorFlow
from __future__ import print_function

import numpy as np
import tensorflow as tf
from mnist_reader import load_mnist
import os
import matplotlib.pyplot as plt
os.system('clear')


# loading the data
path = './data/fashion'
train_images,train_labels = load_mnist(path, kind='train')
test_images,test_labels = load_mnist(path, kind='t10k')


print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)
