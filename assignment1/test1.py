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


import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the L-2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i,j] = np.sqrt(np.sum((X[i]-self.X_train[j])**2))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the L-2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      diff = X[i] - self.X_train
      dists[i,:] = np.sqrt(np.sum(diff**2, axis=1))
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the L-2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the L-2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    x2 = np.sum(X**2, axis=1).reshape((num_test, 1))
    y2 = np.sum(self.X_train**2, axis=1).reshape((1, num_train))
    xy = X.dot(self.X_train.T) # shape is (numtest, numtrain)
    dists = np.sqrt(x2 + y2 - 2*xy) # shape is (numtest, numtrain)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.    
      # np.argsort(dists)
      # closest_y= self.train_y[np.argsort(dists)] # get top k of this
      # get most freq occurrance in closest_y
      #########################################################################
      closest_y = self.y_train[np.argsort(dists[i])][0:k]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = np.argmax(np.bincount(closest_y))
      #sorted_cy = list(closest_y[np.argsort(closest_y[0:k])])
      #most_freq_count = 0
      #most_freq = sorted_cy[0]
      #for x in range(k):   # from 1 - k
      #  count = 0    # start a new counter
      #  tester = sorted_cy[x] # start a new tester
      #  while x < len(sorted_cy) and sorted_cy[x] == tester:   # while sc[x] is the tester
      #      count = count + 1
      #      x = x + 1
      #  if count > most_freq_count:   # if the count is max save it
      #      most_freq_count = count
      #      most_freq = tester
            #print("Most freq so far: ", tester, " with count ", count)
            
      #print()
      #y_pred[i] = most_freq # this should be the most freq in closest_y
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred




    