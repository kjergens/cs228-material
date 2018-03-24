import numpy as np
from cs231n.data_utils import load_CIFAR10

numtest = 500
numtrain = 5000

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
xtrain, ytrain, x, ytest = load_CIFAR10(cifar10_dir)

xtrain = np.reshape(xtrain, (xtrain.shape[0], -1))
x = np.reshape(x, (x.shape[0], -1))

x = x[0:numtest,:]
xtrain = xtrain[0:numtrain,:]
print("x.shape: ", x.shape)
print("xtrain.shape: ", xtrain.shape)

 # HINT: Try to formulate the L-2 distance using 
 #    matrix (dot) multiplication and two broadcast sums.

# Reshape xtrain to be 500 identical rows of 5000 cols, 3072 deep
#x_ = np.ones((numtest, numtrain, 3072))
#print("x_.shape: ", x_.shape)

#xtrain = np.dot(x_, xtrain) 
#print("xtrain.shape: ", xtrain.shape)

# Euclidean dist part 1
# diffs = (np.reshape(x,(numtest, 1, 3072))-xtrain)
# print("diffs.shape: ", diffs.shape)

# temp = diffs**2
# print("squared temp.shape: ", temp.shape)

# # Euclidean dist part 2
# dists = np.sqrt(np.sum(temp, axis=2))

#dists = np.sqrt(((np.reshape(x,(numtest, 1, 3072)) - xtrain)**2).sum(-1))

# Hey Katie, try this when you're not so hungry:
# https://stackoverflow.com/questions/32856726/memory-efficient-l2-norm-using-python-broadcasting
print("x.shape: ", x.shape)
print("xtrain.shape: ", xtrain.shape)


x2 = np.sum(x**2, axis=1).reshape((numtest, 1))
print("x2.shape: ", x2.shape)

y2 = np.sum(xtrain**2, axis=1).reshape((1, numtrain))
print("y2.shape: ", y2.shape)

xy = x.dot(xtrain.T) # shape is (m, n)
print("xy.shape: ", xy.shape)

dists = np.sqrt(x2 + y2 - 2*xy) # shape is (m, n)
print("dists.shape: ", dists.shape)


#dists = np.sqrt(np.dot(diffs[0,0].T, diffs[0,0]))

# show the first dist for the first image
print("the first dist of the first pic: :", dists[0,0])