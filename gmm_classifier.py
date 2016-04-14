"""
==================
GMM classification
==================

Demonstration of Gaussian mixture models for classification.
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from sklearn.mixture import GMM

import common_tools

data_file_prefix = "dataset-3rel_components-quantized_butterfly-2014-12-27"
filename = "./InputData/" + data_file_prefix + ".csv"
selected_features = None  # None will work with all the features. Example of selected features [0, 4, 7, 8] <--- a list of indices
features_data, classifications = common_tools.read_data(filename, delimiter=",", feature_indices=selected_features, num_components=10, dtype='uint8', use_string_labels=False, given_string_labels=True, shuffle_rows=True, has_header=False)

# Partition the data in order to train the classifier
percentage_to_train = 50
num_instances = len(classifications)
holdout_number = int(num_instances * percentage_to_train / 100.)  # hold out percentage (%) of the dataset instances for testing
X_train = features_data[:holdout_number]
y_train = classifications[:holdout_number]
X_test = features_data[holdout_number:]
y_test = classifications[holdout_number:]


n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    print('%d) Using GMM %s' % (index + 1, name))

    t_start = time.clock()

    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)
    t_end = time.clock()
    t_elapsed = t_end - t_start
    print('Building model took: %.9f seconds' % t_elapsed)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print('Train accuracy using mean: %.1f' % train_accuracy)

    y_test_pred = classifier.predict(X_test)
    y_test_true = y_test.reshape(*y_test_pred.shape)

    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print('Test accuracy using mean: %.1f' % test_accuracy)

    errors = y_test_pred - y_test_true
    N = len(y_test_pred)
    correct_count = N - np.count_nonzero(errors)
    prob_correct = correct_count / float(N)

    print("FINAL Test Accuracy = %.2f%s" % (prob_correct * 100., "%"))
    print("-"*80)

