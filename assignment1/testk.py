import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import KNearestNeighbor

# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
# X_train first dim is the number of images (5000)
# Total values is 5000 * (32 x 32 * 3 subpixels) = 5000 * 3072
# If you reshape to 5000, -1, it figures out that the other dimension len should be total/5000 = 3072
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

################
# START HERE
###############

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {k: np.zeros(num_folds) for k in k_choices}

for k in k_to_accuracies:
	# Training phase
    for i in range(num_folds):
    	X_train = X_train_folds[i]
    	y_test_pred = classifier.predict(X_test, k=k, num_loops=0)
        #print("y_test_pred[0:4]: ", y_test_pred[0:4])
        #print("y_train_folds[",i,",0:4]: ", y_train_folds[i][0:4])
    	num_correct = np.sum(y_test_pred == y_test)
    	print("k: ", k, " num_correct: ", num_correct, " len(y_test_pred): ", len(y_test_pred))
    	accuracy = float(num_correct) / len(y_test_pred)
    	print("accuracy: ", accuracy)
    #k_to_accuracies[k][i] = accuracy
    # Validation phase - need to make this different from training
    # y_val_pred = classifier.predict(X_train_folds[num_folds-1], k=k, num_loops=0)
    # num_correct = np.sum(y_val_pred == y_train_folds[num_folds-1])
    # accuracy = float(num_correct) / len(y_val_pred)
    # k_to_accuracies[k][num_folds-1] = accuracy
    # print("k_to_accuracies[",k,"]: ", k_to_accuracies[k])


# Print out the computed accuracies
# for k in sorted(k_to_accuracies):
#     for accuracy in k_to_accuracies[k]:
#         print('k = %d, accuracy = %f' % (k, accuracy))