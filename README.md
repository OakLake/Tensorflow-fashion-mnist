# Tensorflow-fashion-mnist
Getting to know DL better with practice using the fashion-MNIST dataset.
More compute needed, results are not record breaking but it works !

# Data structure
- 60k Training
- 5k Validation
- 5k Testing

# Convnet Structure
- 5x5x32 convolution
- max pool, k=2
- relu
- 5x5x64 convolution
- max pool. k=2
- relu
- fully connected, 128 nodes
- output, 10 nodes, for 10 classes

# Training
- Gradient Descent optimizer
- learning rate decay
- dropout on fc layer (0.8)
- minibatch
