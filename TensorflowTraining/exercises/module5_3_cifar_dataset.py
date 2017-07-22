# Module 3: Datasets
# CIFAR-10 dataset

# Step 1 Get Data
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()

# Step 2 Shuffle Data
from tflearn.data_utils import shuffle, to_categorical
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

