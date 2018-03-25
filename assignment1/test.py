import numpy as np
from cs231n.data_utils import load_CIFAR10

numtest = 50
numtrain = 500

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
xtrain, ytrain, x, ytest = load_CIFAR10(cifar10_dir)

xtrain = np.reshape(xtrain, (xtrain.shape[0], -1))
x = np.reshape(x, (x.shape[0], -1))

x = x[0:numtest,:]
xtrain = xtrain[0:numtrain,:]

 # HINT: Try to formulate the L-2 distance using 
 #    matrix (dot) multiplication and two broadcast sums.
 # The key was to make each matrix (X, X_train) smaller by summing the squares
 #  and then combining them with a dot product.
x2 = np.sum(x**2, axis=1).reshape((numtest, 1))
y2 = np.sum(xtrain**2, axis=1).reshape((1, numtrain))
xy = x.dot(xtrain.T) # shape is (m, n)
dists = np.sqrt(x2 + y2 - 2*xy) # shape is (m, n)


# show the first dist for the first image
print("the first dist of the first pic: :", dists[0,0])

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(xtrain, num_folds)
y_train_folds = np.array_split(ytrain, num_folds)

k_to_accuracies = {k: list(range(num_folds)) for k in k_choices}

print("k_to_accuracies[1]: ", k_to_accuracies[1])

for k in k_to_accuracies:
    for x in k_to_accuracies[k]:
        k_to_accuracies[k] = classifier.predict_labels(np.array(x), k=k)





